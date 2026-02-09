interface TimelineGraphicProps {
  weeksUntilDecision?: number
}

export default function TimelineGraphic({
  weeksUntilDecision,
}: TimelineGraphicProps) {
  const totalWidth = 100 // percentage
  // Position the "Expected Decision" based on weeks relative to a rough timeline
  // Application submitted (left), Decision (calculated), Deadline June 1 (right)
  const decisionPct = weeksUntilDecision !== undefined
    ? Math.max(20, Math.min(80, 30 + weeksUntilDecision * 2))
    : 50

  return (
    <div className="w-full px-2 py-4">
      <svg
        viewBox={`0 0 ${totalWidth} 40`}
        className="w-full h-auto"
        preserveAspectRatio="xMidYMid meet"
        style={{ maxHeight: '80px' }}
      >
        {/* Connecting line */}
        <line
          x1="10"
          y1="16"
          x2="90"
          y2="16"
          stroke="#374151"
          strokeWidth="0.8"
        />

        {/* Filled segment from Application to Decision */}
        <line
          x1="10"
          y1="16"
          x2={decisionPct}
          y2="16"
          stroke="#10b981"
          strokeWidth="0.8"
          opacity="0.5"
        />

        {/* Application dot (filled emerald) */}
        <circle cx="10" cy="16" r="2.5" fill="#10b981" />
        <text
          x="10"
          y="28"
          textAnchor="middle"
          fill="#d1d5db"
          fontSize="3.5"
          fontWeight="500"
        >
          Application
        </text>

        {/* Expected Decision dot (empty emerald ring) */}
        <circle
          cx={decisionPct}
          cy="16"
          r="2.5"
          fill="#050505"
          stroke="#10b981"
          strokeWidth="0.8"
        />
        <text
          x={decisionPct}
          y="8"
          textAnchor="middle"
          fill="#10b981"
          fontSize="3"
          fontWeight="600"
        >
          {weeksUntilDecision !== undefined
            ? `~${weeksUntilDecision} weeks`
            : 'TBD'}
        </text>
        <text
          x={decisionPct}
          y="28"
          textAnchor="middle"
          fill="#d1d5db"
          fontSize="3.5"
          fontWeight="500"
        >
          Expected Decision
        </text>

        {/* Deadline dot (gray) */}
        <circle cx="90" cy="16" r="2.5" fill="#4b5563" />
        <text
          x="90"
          y="28"
          textAnchor="middle"
          fill="#6b7280"
          fontSize="3.5"
          fontWeight="500"
        >
          Deadline June 1
        </text>
      </svg>
    </div>
  )
}
