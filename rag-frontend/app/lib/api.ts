export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function apiFetch(path: string, token: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers);
  headers.set('Authorization', `Bearer ${token}`);
  return fetch(`${API_URL}${path}`, { ...options, headers });
}

export async function apiError(response: Response, fallback: string): Promise<Error> {
  try {
    const data = await response.json();
    const detail = data.detail;
    return new Error(
      typeof detail === 'object' && detail !== null
        ? detail.message || fallback
        : detail || fallback
    );
  } catch {
    return new Error(fallback);
  }
}
