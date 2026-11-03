import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto max-w-xl p-10">
      <h1 className="text-2xl font-semibold">Blue Bikes</h1>
      <p className="mt-2 text-muted-foreground">Bluebikes station tracker demo.</p>
      <Link href="/bikes" className="mt-6 inline-block rounded-md bg-foreground px-4 py-2 text-background">
        Open the Bluebikes map →
      </Link>
    </main>
  );
}
