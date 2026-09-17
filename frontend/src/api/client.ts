import { useQuery } from "@tanstack/react-query";
import type { Analysis, AnalyzeRequest } from "./types";

export class ApiError extends Error {}

async function post(body: AnalyzeRequest): Promise<Analysis> {
  let res: Response;
  try {
    res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "Can't reach the analytics service. Start it with `uvicorn api:app --port 8000`, or append ?mock=1 to browse sample data.",
    );
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `The service returned ${res.status}.`);
  }
  return res.json();
}

/** `?mock=1` (or VITE_MOCK=1) serves the bundled sample analysis. */
export function useAnalysis(body: AnalyzeRequest | null, useMock: boolean) {
  return useQuery({
    queryKey: ["analysis", body, useMock],
    enabled: body !== null,
    queryFn: async () => {
      if (useMock) {
        const mock = (await import("./mock.json")).default;   // dev fixture, loaded on demand
        return mock as unknown as Analysis;
      }
      return post(body!);
    },
  });
}
