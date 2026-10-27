import { BikesClient } from "@/components/BikesClient";
import { TopBar } from "@/components/TopBar";

export default function BikesPage() {
  return (
    <div className="grid h-dvh grid-rows-[auto_1fr] overflow-hidden">
      <TopBar />
      <main aria-label="Bluebikes station tracker" className="relative min-h-0 bg-muted">
        <BikesClient />
      </main>
    </div>
  );
}
