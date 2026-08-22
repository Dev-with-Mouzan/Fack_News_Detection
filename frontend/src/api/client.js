// Thin client for the FastAPI backend.
// Uses relative URLs: Vite proxies /api in dev; FastAPI serves the built
// app same-origin in production. Override with VITE_API_BASE if needed.

const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

/**
 * @param {string} path endpoint path
 * @param {string} text article text
 * @param {{provider?: "gpt"|"google", apiKey?: string}} [aiOpts]
 *   When an API key is set in the app's settings it is forwarded to the
 *   backend so AI verification uses the user's own account.
 */
async function post(path, text, aiOpts = {}) {
  const body = { text };
  if (aiOpts.apiKey) {
    body.provider = aiOpts.provider || "gpt";
    body.api_key = aiOpts.apiKey;
  }

  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Cannot reach the server. Make sure the backend is running on port 8000.");
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }

  return res.json();
}

export const predictML = (text) => post("/predict/ml", text);
export const predictAI = (text, aiOpts) => post("/predict/ai", text, aiOpts);
export const predictCombined = (text, aiOpts) => post("/predict", text, aiOpts);
