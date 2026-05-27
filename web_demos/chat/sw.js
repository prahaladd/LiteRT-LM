const CACHE_NAME = 'litertlm-chat-shell-v1';

self.addEventListener('install', (e) => {
  // Claim client instantly without waiting for refresh
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(clients.claim());
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  
  // Only intercept GET requests
  if (e.request.method !== 'GET') {
    return;
  }

  // 1. STRICTLY IGNORE giant model weight files (.litertlm)
  // Caching gigabytes inside Service Worker Cache causes performance bottlenecks and storage exhaustion.
  // Model caching is already managed efficiently in main.ts inside a distinct Chrome Cache Storage bucket.
  if (url.pathname.endsWith('.litertlm')) {
    return;
  }

  // 2. Intercept local UI shell requests with Network-First strategy
  // Prioritizes network fetches so developers and users receive updates instantly on reload,
  // falling back strictly to local shell caches when offline.
  e.respondWith(
    fetch(e.request, { credentials: 'include' })
      .then((response) => {
        // Cache local origin resources dynamically on the fly
        if (response.status === 200 && url.origin === self.location.origin) {
          const cacheCopy = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(e.request, cacheCopy);
          });
        }
        return response;
      })
      .catch((err) => {
        console.log('[PWA SW] Network failed or offline, falling back to cache for:', url.pathname);
        return caches.match(e.request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          console.error('[PWA SW] Network fetch failed and no cache match:', err);
          return new Response('Network error occurred.', { status: 408, headers: { 'Content-Type': 'text/plain' } });
        });
      })
  );
});
