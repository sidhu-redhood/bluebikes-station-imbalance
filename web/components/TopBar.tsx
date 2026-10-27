import Link from "next/link";

/** Thin top navigation bar shared across CoolBoston pages. */
export function TopBar() {
  return (
    <header className="flex items-center gap-4 border-b border-border bg-card px-4 py-2">
      <Link href="/" className="font-semibold">
        CoolBoston
      </Link>
      <nav className="text-sm text-muted-foreground">
        <span aria-current="page" className="text-foreground">
          Bluebikes tracker
        </span>
      </nav>
    </header>
  );
}
