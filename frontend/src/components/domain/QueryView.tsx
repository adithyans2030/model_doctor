import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { ErrorState, LoadingRegion } from "@/components/ui";

/**
 * Renders the three states every data view needs: a labelled skeleton while loading,
 * an error with retry, and (optionally) an empty state; otherwise the content.
 */
export function QueryView<T>({ query, label, skeleton, isEmpty, empty, children }: {
  query: UseQueryResult<T>;
  /** What is loading, for screen readers: "Loading models". */
  label: string;
  skeleton: ReactNode;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <LoadingRegion label={label}>{skeleton}</LoadingRegion>;
  if (query.isError) {
    return <ErrorState compact title={`${label.replace(/^Loading /, "Couldn't load ")}`} error={query.error} onRetry={() => query.refetch()} />;
  }
  if (isEmpty?.(query.data) && empty) return <>{empty}</>;
  return <>{children(query.data)}</>;
}
