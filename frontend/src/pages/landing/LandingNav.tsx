import { Menu, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { GithubMark } from "@/components/GithubMark";
import { Logo } from "@/components/Logo";
import { ThemeToggle } from "@/components/ThemeToggle";
import { buttonClasses } from "@/components/ui";
import { DOCS, GITHUB } from "./content";

const LINKS = [
  { href: "#features", label: "Features" },
  { href: "#how-it-works", label: "How it works" },
  { href: "#demo", label: "Demo" },
  { href: DOCS, label: "Docs", external: true },
];

export function LandingNav() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-bg/80 backdrop-blur supports-[backdrop-filter]:bg-bg/65">
      <nav aria-label="Main" className="mx-auto flex h-14 max-w-[1200px] items-center gap-6 px-4 sm:px-6">
        <Link to="/" className="no-underline hover:no-underline" aria-label="ModelDoctor home"><Logo /></Link>
        <ul className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <li key={l.label}>
              <a href={l.href} {...(l.external ? { target: "_blank", rel: "noreferrer" } : {})}
                className="rounded-md px-2.5 py-1.5 text-sm text-fg-2 no-underline hover:bg-surface-2 hover:text-fg hover:no-underline">
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <span className="flex-1" />
        <ThemeToggle />
        <a href={GITHUB} target="_blank" rel="noreferrer" className={buttonClasses({ variant: "ghost", size: "sm", className: "max-sm:hidden" })}>
          <GithubMark size={15} /> GitHub
        </a>
        <Link to="/app" className={buttonClasses({ variant: "primary", size: "sm", className: "max-sm:hidden" })}>Try the demo</Link>
        <button type="button" className={buttonClasses({ variant: "ghost", size: "sm", className: "w-8 px-0 md:hidden" })}
          aria-expanded={open} aria-controls="mobile-nav" aria-label={open ? "Close menu" : "Open menu"} onClick={() => setOpen((o) => !o)}>
          {open ? <X size={16} aria-hidden /> : <Menu size={16} aria-hidden />}
        </button>
      </nav>
      {open && (
        <div id="mobile-nav" className="border-t border-line bg-bg px-4 py-3 md:hidden">
          <ul className="space-y-1">
            {LINKS.map((l) => (
              <li key={l.label}>
                <a href={l.href} onClick={() => setOpen(false)} {...(l.external ? { target: "_blank", rel: "noreferrer" } : {})}
                  className="block rounded-md px-2 py-2 text-sm text-fg-2 no-underline hover:bg-surface-2 hover:text-fg hover:no-underline">{l.label}</a>
              </li>
            ))}
            <li><a href={GITHUB} target="_blank" rel="noreferrer" className="block rounded-md px-2 py-2 text-sm text-fg-2 no-underline hover:bg-surface-2 hover:no-underline">GitHub</a></li>
          </ul>
          <Link to="/app" className={buttonClasses({ variant: "primary", className: "mt-2 w-full" })}>Try the demo</Link>
        </div>
      )}
    </header>
  );
}
