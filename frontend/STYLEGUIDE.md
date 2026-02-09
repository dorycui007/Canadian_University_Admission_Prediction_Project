## University Admission Project Frontend Design System

This guide documents the University Admission Project's public-facing design language. It evolves our previous minimal aesthetic into a bolder, **editorial-first** identity featuring deep atmospheric backgrounds and high-contrast typography.

### Design Goals

- **Style Inspiration**: **Bold, Editorial, Atmospheric**. Deep backgrounds with diffuse lighting; high-contrast white text; mixing geometric sans-serif with elegant italic serifs for narrative emphasis.
- **Clarity & Impact**: Marketing surfaces use heavy weights and "poster-style" layouts. App surfaces remain high-signal and low-noise for data density.
- **Coherence**: Consistent glass surfaces (`GlassCard`), shared layouts, and strict spacing rules.
- **Motion with Restraint**: Subtle transitions and micro-interactions; respect `prefers-reduced-motion`.
- **A11y First**: Semantic HTML, keyboard focus, and high contrast text (White on Black).

### Tech Stack & Conventions

- **React + TypeScript** with **Vite**.
- **Tailwind CSS** for utilities.
- **shadcn/ui + Radix** for primitives (buttons, tabs, slider, dialog).
- **Animations**: `animejs` for timelines. **Do not use GSAP**.

#### Width Policy

- **Public/Marketing Pages**:
  - **Fixed width** for readability. Use `container mx-auto px-6`.
  - **Poster/Hero Sections**: Can be full-height (`min-h-screen`) but content remains constrained to `max-w-6xl`.
- **App/Dashboard Pages**:
  - Full-width layouts allowed. Use `px-6` padding without `container` constraint to keep grids breathable.

### Visual Identity

- **Background**: Deep, rich dark surfaces.
  - Base: `bg-black` or `bg-[#050505]`.
  - **Atmosphere**: Use large, diffuse **Emerald** radial gradients ("Orbs") behind text to create depth. Avoid flat tints.
- **Color Usage**:
  - **Primary Accent**: **Emerald** (Success/Growth). Used for "current state," positive assertions, and primary glows.
  - **Secondary**: White (Text/Borders), Gray-500 (Subtext).
  - **Functional**: Amber (Warn), Red (Critical), Blue (Neutral).
- **Typography (The "Editorial" Mix)**:
  - **Primary (Sans)**: **Mona Sans**. Used for headlines (Bold/Heavy in marketing, Light in app) and body copy.
  - **Accent (Serif)**: **Newsreader (Italic)** or similar high-contrast serif. Used for emphasis phrases (e.g., *"ours isn't"*, *"now"*).
  - **Hierarchy**:
    - **Marketing**: Heavy, tight tracking `tracking-tight`, solid white.
    - **App**: Light/Regular weights, tabular figures for metrics.

### Tokens & Utilities

- **Neutrals on Dark**:
  - Surface: `bg-white/5` (Glass), `bg-black` (Base).
  - Borders: `border-white/10`.
  - Text: `text-white` (Primary), `text-gray-400` (Secondary).
- **Radius**:
  - Cards: `rounded-2xl`.
  - Buttons/Pills: `rounded-full`.
- **Glow Utilities**:
  - `bg-emerald-glow`: A radial gradient helper (e.g., `radial-gradient(circle at center, rgba(16, 185, 129, 0.15) 0%, transparent 70%)`).
- **Scrollbars**:
  - Always use `custom-scrollbar` with `overflow-y-auto`.

### Core Components (Shared)

#### SectionHeader (Marketing)

Standard block for marketing sections.

```tsx
<SectionHeader
  label="Admissions"
  title="Your future starts now."
  description="We built the tool universities actually want you to use."
/>