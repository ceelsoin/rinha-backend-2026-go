//go:build linux

// lb — FD-passing load balancer for Rinha de Backend 2026.
//
// Accepts TCP connections on port 9999 and hands off each client file
// descriptor to one of the API workers via SCM_RIGHTS over a Unix DGRAM
// socket (round-robin). The worker receives the raw TCP fd and handles the
// full HTTP request/response without any proxy intermediary — eliminating the
// double-copy overhead of a traditional reverse proxy (nginx).
//
// Environment variables:
//   UPSTREAMS   comma-separated Unix socket paths for API workers
//               (default: /sockets/api1.sock,/sockets/api2.sock)

package main

import (
	"log"
	"os"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// SO_REUSEPORT is not exported by syscall on all Go/Linux builds;
// hardcode the Linux value (0xf = 15) which is stable across all archs.
const soReusePort = 0xf

func main() {
	log.SetFlags(log.Ltime | log.Lmsgprefix)

	upstreamsEnv := os.Getenv("UPSTREAMS")
	if upstreamsEnv == "" {
		upstreamsEnv = "/sockets/api1.sock,/sockets/api2.sock"
	}
	var upstreams []string
	for _, p := range strings.Split(upstreamsEnv, ",") {
		if p = strings.TrimSpace(p); p != "" {
			upstreams = append(upstreams, p)
		}
	}
	if len(upstreams) == 0 {
		log.Fatal("[lb] no upstreams configured")
	}

	// Pin to a single OS thread — the accept loop is single-threaded.
	runtime.GOMAXPROCS(1)

	// ── TCP listener on :9999 ────────────────────────────────────────────────
	listenFd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		log.Fatalf("[lb] socket: %v", err)
	}
	if err := syscall.SetsockoptInt(listenFd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1); err != nil {
		log.Fatalf("[lb] SO_REUSEADDR: %v", err)
	}
	if err := syscall.SetsockoptInt(listenFd, syscall.SOL_SOCKET, soReusePort, 1); err != nil {
		log.Fatalf("[lb] SO_REUSEPORT: %v", err)
	}
	sa := &syscall.SockaddrInet4{Port: 9999}
	if err := syscall.Bind(listenFd, sa); err != nil {
		log.Fatalf("[lb] bind :9999: %v", err)
	}
	if err := syscall.Listen(listenFd, 8192); err != nil {
		log.Fatalf("[lb] listen: %v", err)
	}

	// ── Unix DGRAM socket for sending fds (SCM_RIGHTS) ───────────────────────
	// DGRAM: connectionless — we send to each upstream path directly.
	udsFd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_DGRAM, 0)
	if err != nil {
		log.Fatalf("[lb] unix socket: %v", err)
	}
	// Large send buffer so bursts don't block even if an API is momentarily slow.
	syscall.SetsockoptInt(udsFd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, 16*1024*1024) //nolint

	// ── Wait for all upstream sockets to be ready (up to 10 s) ──────────────
	// The API containers may take a moment to bind their UNIX sockets after
	// the LB container starts (depends_on only waits for container start, not
	// application readiness). Accepting connections before the sockets exist
	// causes sendFd to fail with ENOENT → spurious 503 responses.
	seen := make(map[string]bool, len(upstreams))
	deadline := time.Now().Add(10 * time.Second)
	for len(seen) < len(upstreams) && time.Now().Before(deadline) {
		for _, path := range upstreams {
			if seen[path] {
				continue
			}
			if _, err := os.Stat(path); err == nil {
				seen[path] = true
				log.Printf("[lb] upstream ready: %s", path)
			}
		}
		if len(seen) < len(upstreams) {
			time.Sleep(5 * time.Millisecond)
		}
	}
	if len(seen) < len(upstreams) {
		log.Printf("[lb] warning: not all upstream sockets appeared within 10s")
	}

	log.Printf("[lb] listening on :9999, upstreams=%v", upstreams)

	nUp := uint64(len(upstreams))
	var rr uint64

	for {
		// Accept4: atomically set SOCK_CLOEXEC so the fd doesn't leak on exec.
		clientFd, _, err := syscall.Accept4(listenFd, syscall.SOCK_CLOEXEC)
		if err != nil {
			continue
		}

		// TCP_NODELAY: send small HTTP responses immediately without Nagle delay.
		syscall.SetsockoptInt(clientFd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, 1) //nolint

		// Round-robin select upstream.
		path := upstreams[rr%nUp]
		rr++

		// Send fd to the chosen API worker via SCM_RIGHTS.
		// Retry a few times to handle transient ENOBUFS or late socket binding.
		// On persistent failure, just close — the LB must not send any response
		// before passing the connection (per competition rules: LB applies no logic).
		if err := sendFd(udsFd, path, clientFd); err != nil {
			log.Printf("[lb] sendFd failed: %v", err)
		}

		// LB closes its copy — the worker received a dup'd fd from the kernel.
		syscall.Close(clientFd) //nolint
	}
}

// sendFd passes clientFd to the Unix DGRAM socket at path using SCM_RIGHTS.
// Retries up to 5 times with 1 ms backoff before giving up.
func sendFd(udsFd int, path string, clientFd int) error {
	rights := syscall.UnixRights(clientFd)
	dummy := []byte{1} // non-empty iov required by sendmsg
	addr := &syscall.SockaddrUnix{Name: path}
	var err error
	for i := 0; i < 5; i++ {
		if err = syscall.Sendmsg(udsFd, dummy, rights, addr, syscall.MSG_NOSIGNAL); err == nil {
			return nil
		}
		time.Sleep(1 * time.Millisecond)
	}
	return err
}

const resp503 = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"

func send503(fd int) {
	syscall.Write(fd, []byte(resp503)) //nolint
}

