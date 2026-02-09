export type PredictionLabel = "LIKELY_ADMIT" | "UNCERTAIN" | "UNLIKELY_ADMIT"

export interface ConfidenceInterval {
  lower: number
  upper: number
  method: string
}

export interface FeatureImportance {
  feature_name: string
  value: number
  coefficient: number
  contribution: number
  direction: "+" | "-"
}

export interface SimilarProgram {
  university: string
  program: string
  similarity: number
  historical_admit_rate: number | null
}

export interface Activity {
  description: string
}

export interface ECAssessment {
  score: number
  tier: number
  tier_label: string
  category_breakdown: Array<{ category: string; active: boolean }>
  activity_tiers: number[]
}

export interface ApplicationRequest {
  top_6_average: number
  university: string
  program: string
  grade_11_average?: number
  grade_12_average?: number
  province?: string
  country?: string
  application_year?: number
  activities?: Activity[]
}

export interface PredictionResponse {
  probability: number
  confidence_interval: ConfidenceInterval
  prediction: string
  feature_importance: FeatureImportance[]
  similar_programs: SimilarProgram[]
  model_version: string
  timestamp: string
  calibration_note?: string
  warnings: string[]
  ec_assessment?: ECAssessment
  similar_cases?: SimilarCase[]
}

export interface BatchPredictionRequest {
  applications: ApplicationRequest[]
  return_similar_programs: boolean
  return_feature_importance: boolean
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[]
  total_count: number
  success_count: number
  error_count: number
  errors: Array<{ index: number; error: string }>
  processing_time_ms: number
}

export interface ModelInfo {
  version: string
  training_date: string
  training_samples: number
  feature_count: number
  universities_supported: number
  programs_supported: number
  metrics: Record<string, number>
  calibration_method: string
  embedding_dim: number
}

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy"
  model_loaded: boolean
  database_connected: boolean
  embedding_service_available: boolean
  timestamp: string
  uptime_seconds: number
}

export type GradeFormat = "percentage" | "gpa4" | "gpa43" | "ib45" | "letter"

export interface SimilarCase {
  grade: number
  ec_score: number
  ec_tier: number
  outcome: "accepted" | "rejected" | "waitlisted"
  university: string
  program: string
}

export interface DistributionData {
  bins: number[]
  counts_accepted: number[]
  counts_rejected: number[]
  statistics: {
    mean: number
    median: number
    p25: number
    p75: number
    min: number
    max: number
    n: number
  }
  year_trend?: Array<{ year: number; median_grade: number }>
}

// --- Program Analytics types ---

export type ConfidenceLevel = "low" | "moderate" | "high"

export type DifficultyLabel =
  | "Very Competitive"
  | "Competitive"
  | "Moderate"
  | "Accessible"
  | "Unknown"

export interface AdmittedGradeRange {
  min: number
  p25: number
  median: number
  p75: number
  max: number
  n: number
}

export interface CompetitivenessByYear {
  year: string
  admitted_range: AdmittedGradeRange | null
  total_reports: number
  confidence_level: ConfidenceLevel
}

export interface GradeStats {
  mean: number
  median: number
  p25: number
  p75: number
  min: number
  max: number
  std: number
  n: number
}

export interface YearTrendPoint {
  year: number
  median_grade: number
  median_accepted: number
  count: number
}

export interface DataQuality {
  missing_grade_pct: number
  earliest_cycle: string | null
  latest_cycle: string | null
  sparse_warning: string | null
}

export interface OfferByMonth {
  month: string
  count: number
}

export interface OfferTimeline {
  earliest_month?: string
  median_month?: string
  latest_month?: string
  by_month: OfferByMonth[]
  total_with_dates: number
}

export interface GradeProgression {
  g11_avg?: number
  g11_median?: number
  g11_n?: number
  g12_midterm_avg?: number
  g12_midterm_median?: number
  g12_midterm_n?: number
  g12_final_avg?: number
  g12_final_median?: number
  g12_final_n?: number
  n: number
}

export interface ApplicantTypeBreakdown {
  "101": number
  "105": number
  unknown: number
}

export interface ProgramAnalytics {
  university: string
  program: string
  total_records: number
  cycle_years: string[]
  competitiveness: {
    difficulty: DifficultyLabel
    admitted_range: AdmittedGradeRange | null
    sample_size: number
    confidence_level: ConfidenceLevel
    by_year: CompetitivenessByYear[]
  }
  decision_breakdown: Record<string, number>
  grade_statistics: {
    all: GradeStats | null
    accepted: GradeStats | null
    rejected: GradeStats | null
  }
  distribution: DistributionData
  year_trend: YearTrendPoint[]
  offer_timeline: OfferTimeline
  grade_progression: GradeProgression | null
  applicant_type: ApplicantTypeBreakdown
  province_breakdown: Record<string, number>
  data_quality: DataQuality
}

export interface ProgramListingEntry {
  university: string
  program: string
  total_records: number
  difficulty: DifficultyLabel
  median_grade_accepted: number | null
  confidence_level: ConfidenceLevel
  avg_grade_accepted: number
}
