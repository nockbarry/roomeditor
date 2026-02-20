import { api } from "./client.ts";
import type { BenchmarksResponse } from "../types/api.ts";

export async function getBenchmarks(): Promise<BenchmarksResponse> {
  return api.get<BenchmarksResponse>("/benchmarks");
}
