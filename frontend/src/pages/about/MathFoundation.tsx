import GlassCard from '@/components/shared/GlassCard'
import SectionHeader from '@/components/shared/SectionHeader'
import Breadcrumb from '@/components/shared/Breadcrumb'
import PrevNextNav from '@/components/shared/PrevNextNav'
import TableOfContents, { type TocItem } from '@/components/shared/TableOfContents'
import DependencyChainDiagram from '@/components/interactive/DependencyChainDiagram'
import SVDShrinkageExplorer from '@/components/interactive/SVDShrinkageExplorer'
import RidgePathExplorer from '@/components/interactive/RidgePathExplorer'
import DotProductVisualizer from '@/components/interactive/DotProductVisualizer'
import HouseholderStepThrough from '@/components/interactive/HouseholderStepThrough'
import ProjectionVisualizer from '@/components/interactive/ProjectionVisualizer'
import M from '@/components/shared/Math'

/* ─────────────────────────────────────────────
   Table of contents items
   ───────────────────────────────────────────── */
const tocItems: TocItem[] = [
  { id: 'chain', label: 'Dependency Chain' },
  { id: 'vectors', label: 'Vectors' },
  { id: 'matrices', label: 'Matrices' },
  { id: 'projections', label: 'Projections' },
  { id: 'qr', label: 'QR Factorization' },
  { id: 'svd', label: 'SVD' },
  { id: 'ridge', label: 'Ridge Regression' },
]

/* ─────────────────────────────────────────────
   Vector functions table data
   ───────────────────────────────────────────── */
const VECTOR_FUNCTIONS = [
  { fn: 'dot(u, v)', formula: '\\sum u_i v_i', use: 'Core of all predictions' },
  { fn: 'norm(v)', formula: '\\sqrt{\\sum v_i^2}', use: 'Distance, normalization' },
  { fn: 'scale(v, c)', formula: 'c \\cdot v', use: 'Scaling vectors' },
  { fn: 'add(u, v)', formula: 'u + v', use: 'Vector addition' },
  { fn: 'subtract(u, v)', formula: 'u - v', use: 'Residuals' },
  { fn: 'normalize(v)', formula: 'v / \\|v\\|', use: 'Unit vectors for projections' },
  { fn: 'cosine_similarity(u, v)', formula: '\\frac{\\text{dot}(u,v)}{\\|u\\| \\cdot \\|v\\|}', use: 'Embedding similarity' },
  { fn: 'angle(u, v)', formula: '\\arccos(\\text{sim})', use: 'Geometric interpretation' },
]

/* ─────────────────────────────────────────────
   Matrix functions table data
   ───────────────────────────────────────────── */
const MATRIX_FUNCTIONS = [
  { fn: 'multiply(A, B)', purpose: 'Matrix multiplication', math: 'X\\beta' },
  { fn: 'transpose(A)', purpose: 'Transpose', math: 'A^T' },
  { fn: 'identity(n)', purpose: 'Ridge regularization', math: 'X^TX + \\lambda I' },
  { fn: 'trace(A)', purpose: 'Effective degrees of freedom', math: '\\sum \\text{diag}(A)' },
  { fn: 'rank(A)', purpose: 'Linearly independent columns', math: null },
  { fn: 'condition_number(A)', purpose: 'Numerical stability', math: '\\sigma_{\\max} / \\sigma_{\\min}' },
]

/* ─────────────────────────────────────────────
   Ridge functions table data
   ───────────────────────────────────────────── */
const RIDGE_FUNCTIONS = [
  { fn: 'ridge_solve', description: 'Standard ridge', math: '\\hat{\\beta} = (X^TX + \\lambda I)^{-1}X^Ty' },
  { fn: 'ridge_solve_qr', description: 'QR-based solver for better numerical stability', math: null },
  { fn: 'weighted_ridge_solve', description: 'Weighted ridge -- used by every IRLS iteration', math: '(X^TWX + \\lambda I)\\beta = X^TWz' },
  { fn: 'ridge_loocv', description: 'Leave-one-out CV via the hat matrix shortcut', math: null },
  { fn: 'ridge_cv', description: 'K-fold cross-validation for \u03BB selection', math: null },
  { fn: 'ridge_path', description: 'Coefficient paths across a grid of \u03BB values', math: null },
  { fn: 'ridge_gcv', description: 'Generalized cross-validation (closed-form)', math: null },
  { fn: 'ridge_effective_df', description: 'Effective degrees of freedom for a given \u03BB', math: '\\text{trace}(H)' },
]

/* ─────────────────────────────────────────────
   Page
   ───────────────────────────────────────────── */
export default function MathFoundation() {
  return (
    <div className="min-h-screen">
      {/* ── Hero ──────────────────────────────── */}
      <section className="pt-24 pb-8">
        <div className="container mx-auto px-6 max-w-6xl">
          <Breadcrumb
            items={[
              { label: 'About', to: '/about' },
              { label: 'Math Foundation' },
            ]}
          />

          <div className="mt-6">
            <SectionHeader
              label="Math Foundation"
              title="Linear algebra from"
              accent="scratch."
              description="Every prediction is a dot product. Here is every operation that makes it work, implemented without ML libraries."
            />
          </div>
        </div>
      </section>

      {/* ── Layout: sidebar + content ─────────── */}
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="flex gap-6 lg:gap-12">
          <TableOfContents items={tocItems} />

          <main className="min-w-0 flex-1">
            {/* ── Dependency Chain ─────────────── */}
            <section id="chain" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Overview"
                  title="The dependency chain."
                />

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  Each module builds on the one before it. Vectors feed into
                  matrices, matrices into projections, and so on up to the
                  logistic regression model that produces the final probability.
                </p>

                <DependencyChainDiagram />
              </div>
            </section>

            {/* ── Vectors ─────────────────────── */}
            <section id="vectors" className="py-16">
              <div className="space-y-10">
                <SectionHeader label="Module 1" title="Vectors." />

                <GlassCard>
                  <p className="text-white font-medium">
                    The dot product is <strong>THE</strong> core operation --
                    every prediction computes{' '}
                    <M>{"z = \\mathbf{x}^T\\boldsymbol{\\beta}"}</M>
                  </p>
                </GlassCard>

                <p className="text-gray-400 leading-relaxed max-w-3xl">
                  All higher-level operations -- matrix multiply, projections,
                  QR factorization -- ultimately decompose into dot products.
                  Getting these right (and fast) is the foundation of
                  everything.
                </p>

                <GlassCard padding="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Function</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Formula</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Use</th>
                        </tr>
                      </thead>
                      <tbody>
                        {VECTOR_FUNCTIONS.map((row) => (
                          <tr
                            key={row.fn}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                              {row.fn}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-emerald-400">
                              <M>{row.formula}</M>
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.use}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>

                <GlassCard>
                  <DotProductVisualizer />
                </GlassCard>
              </div>
            </section>

            {/* ── Matrices ────────────────────── */}
            <section id="matrices" className="py-16">
              <div className="space-y-10">
                <SectionHeader label="Module 2" title="Matrices." />

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    The design matrix{' '}
                    <M>{"X"}</M>{' '}
                    has shape{' '}
                    <M>{"(n_{\\text{samples}} \\times n_{\\text{features}})"}</M>
                    , for example{' '}
                    <M>{"(4900 \\times 250)"}</M>
                    . Each row is one applicant. Each column is one feature
                    (grade z-score, university embedding dimension, province
                    indicator, interaction term).
                  </p>
                  <p>
                    Matrix multiplication{' '}
                    <M>{"X\\beta"}</M>{' '}
                    computes all 4,900 predictions in a single operation --
                    each output element is a dot product of one row with the
                    coefficient vector.
                  </p>
                </div>

                <GlassCard padding="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Function</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Purpose</th>
                        </tr>
                      </thead>
                      <tbody>
                        {MATRIX_FUNCTIONS.map((row) => (
                          <tr
                            key={row.fn}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                              {row.fn}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.purpose}
                              {row.math && (
                                <>
                                  {' -- '}
                                  <M>{row.math}</M>
                                </>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>

                {/* Worked example */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">
                    Worked example: <M>{"3 \\times 3"}</M> multiply
                  </h3>
                  <GlassCard>
                    <pre className="font-mono text-xs sm:text-sm text-gray-300 leading-relaxed overflow-x-auto">
{`A = | 1  2  3 |    B = | 7   8  |
    | 4  5  6 |        | 9  10  |
                       | 11  12 |

C = A @ B

C[0,0] = 1*7  + 2*9  + 3*11  = 58
C[0,1] = 1*8  + 2*10 + 3*12  = 64
C[1,0] = 4*7  + 5*9  + 6*11  = 139
C[1,1] = 4*8  + 5*10 + 6*12  = 154

C = | 58   64  |
    | 139  154 |`}
                    </pre>
                    <p className="mt-4 text-gray-500 text-sm">
                      Every element of C is a dot product of a row of A with a
                      column of B. This is exactly what{' '}
                      <code className="text-emerald-400 font-mono text-sm">
                        multiply(A, B)
                      </code>{' '}
                      computes.
                    </p>
                  </GlassCard>
                </div>
              </div>
            </section>

            {/* ── Projections ─────────────────── */}
            <section id="projections" className="py-16">
              <div className="space-y-10">
                <SectionHeader label="Module 3" title="Projections." />

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    The <strong className="text-white">hat matrix</strong>{' '}
                    projects observed outcomes onto the column space of the
                    design matrix. It "puts a hat on y to get{' '}
                    <M>{"\\hat{y}"}</M>
                    " -- the geometric heart of least squares.
                  </p>
                </div>

                <GlassCard>
                  <div className="space-y-4">
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
                      The Hat Matrix
                    </p>
                    <M display>{"H = X(X^TX)^{-1}X^T"}</M>
                    <p className="text-gray-400 text-sm leading-relaxed">
                      <M>{"\\hat{y} = Hy"}</M>{' '}
                      -- the predicted values are a linear transformation of the
                      observed outcomes.
                    </p>
                  </div>
                </GlassCard>

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    <strong className="text-white">Leverage:</strong>{' '}
                    <M>{"h_{ii}"}</M>{' '}
                    is the diagonal of the hat matrix. It measures how much
                    influence observation i has on its own predicted value. High
                    leverage points can disproportionately affect the regression
                    fit.
                  </p>
                  <p>
                    Leverage values satisfy{' '}
                    <M>{"0 \\le h_{ii} \\le 1"}</M>{' '}
                    and{' '}
                    <M>{"\\sum h_{ii} = p"}</M>{' '}
                    (the number of features). Points with{' '}
                    <M>{"h_{ii} > 2p/n"}</M>{' '}
                    are flagged as high-leverage.
                  </p>
                </div>

                <GlassCard>
                  <ProjectionVisualizer />
                </GlassCard>
              </div>
            </section>

            {/* ── QR Factorization ────────────── */}
            <section id="qr" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Module 4"
                  title="QR Factorization."
                />

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    The normal equations compute{' '}
                    <M>{"X^TX"}</M>
                    , which{' '}
                    <strong className="text-white">
                      squares the condition number
                    </strong>
                    . QR factorization avoids this entirely, giving us much
                    better numerical stability.
                  </p>
                </div>

                {/* Condition number callout */}
                <GlassCard>
                  <div className="text-center space-y-2">
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
                      The Stability Argument
                    </p>
                    <M display>{"\\mathrm{cond}(X^TX) = \\mathrm{cond}(X)^2"}</M>
                    <p className="text-gray-500 text-sm">
                      If <M>{"\\mathrm{cond}(X) = 10^3"}</M>, then{' '}
                      <M>{"\\mathrm{cond}(X^TX) = 10^6"}</M>
                      {' '}-- six orders of magnitude of potential numerical error.
                    </p>
                  </div>
                </GlassCard>

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    <strong className="text-white">
                      Householder reflections:
                    </strong>{' '}
                    each reflection zeros out one column below the diagonal.
                    After n reflections, we have an upper triangular matrix R
                    and an orthogonal matrix Q.
                  </p>
                </div>

                {/* Progressive zeroing visual */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <GlassCard>
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-medium mb-3">
                      Initial X
                    </p>
                    <pre className="font-mono text-xs sm:text-sm text-gray-300 leading-relaxed overflow-x-auto">
{`| x  x  x |
| x  x  x |
| x  x  x |
| x  x  x |`}
                    </pre>
                  </GlassCard>
                  <GlassCard>
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-medium mb-3">
                      After <M>{"H_1"}</M>
                    </p>
                    <pre className="font-mono text-xs sm:text-sm leading-relaxed overflow-x-auto">
{`| `}<span className="text-emerald-400">r  r  r</span>{` |
| `}<span className="text-emerald-400">0</span>{`  x  x |
| `}<span className="text-emerald-400">0</span>{`  x  x |
| `}<span className="text-emerald-400">0</span>{`  x  x |`}
                    </pre>
                  </GlassCard>
                  <GlassCard>
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-medium mb-3">
                      After <M>{"H_2"}</M>
                    </p>
                    <pre className="font-mono text-xs sm:text-sm leading-relaxed overflow-x-auto">
{`| `}<span className="text-emerald-400">r  r  r</span>{` |
| `}<span className="text-emerald-400">0  r  r</span>{` |
| `}<span className="text-emerald-400">0  0</span>{`  x |
| `}<span className="text-emerald-400">0  0</span>{`  x |`}
                    </pre>
                  </GlassCard>
                  <GlassCard>
                    <p className="text-xs uppercase tracking-widest text-gray-500 font-medium mb-3">
                      After <M>{"H_3"}</M>
                    </p>
                    <pre className="font-mono text-xs sm:text-sm text-emerald-400 leading-relaxed overflow-x-auto">
{`| r  r  r |
| 0  r  r |
| 0  0  r |
| 0  0  0 |`}
                    </pre>
                  </GlassCard>
                </div>

                <GlassCard>
                  <HouseholderStepThrough />
                </GlassCard>

                {/* Back substitution */}
                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    <strong className="text-white">Back substitution:</strong>{' '}
                    once we have <M>{"R\\beta = Q^Ty"}</M>, we solve from the bottom
                    row up. The last equation has one unknown, the second-to-last
                    has two (but one is already known), and so on.
                  </p>
                </div>

                {/* Solving least squares */}
                <GlassCard>
                  <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-4">
                    Solving Least Squares via QR
                  </p>
                  <ol className="space-y-3">
                    <li className="flex items-start gap-3">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold shrink-0 mt-0.5">
                        1
                      </span>
                      <span className="text-gray-300 text-sm">
                        Factor{' '}
                        <M>{"X = QR"}</M>{' '}
                        using Householder reflections
                      </span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold shrink-0 mt-0.5">
                        2
                      </span>
                      <span className="text-gray-300 text-sm">
                        Compute{' '}
                        <M>{"c = Q^Ty"}</M>{' '}
                        -- project y onto the orthonormal basis
                      </span>
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-400/10 text-emerald-400 text-xs font-bold shrink-0 mt-0.5">
                        3
                      </span>
                      <span className="text-gray-300 text-sm">
                        Solve{' '}
                        <M>{"R\\beta = c"}</M>{' '}
                        via back substitution
                      </span>
                    </li>
                  </ol>
                </GlassCard>
              </div>
            </section>

            {/* ── SVD ─────────────────────────── */}
            <section id="svd" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Module 5"
                  title="Singular Value Decomposition."
                />

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    Every matrix can be decomposed as{' '}
                    <M>{"A = U\\Sigma V^T"}</M>{' '}
                    where U and V are orthogonal and{' '}
                    <M>{"\\Sigma"}</M> is diagonal with
                    non-negative singular values{' '}
                    <M>{"\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge \\sigma_r \\ge 0"}</M>.
                  </p>
                  <p>
                    The SVD reveals the fundamental geometry of a linear
                    transformation: <M>{"V^T"}</M> rotates,{' '}
                    <M>{"\\Sigma"}</M> scales, and <M>{"U"}</M>{' '}
                    rotates again. The singular values tell us everything about
                    the matrix's numerical behavior.
                  </p>
                </div>

                {/* Four uses grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <GlassCard>
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                      Condition Number
                    </p>
                    <div className="mb-2">
                      <M display>{"\\kappa(A) = \\sigma_1 / \\sigma_r"}</M>
                    </div>
                    <p className="text-gray-500 text-sm">
                      Ratio of largest to smallest singular value. High values
                      mean the matrix is nearly singular and solutions are
                      numerically unstable.
                    </p>
                  </GlassCard>

                  <GlassCard>
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                      Rank Detection
                    </p>
                    <div className="mb-2">
                      <M display>{"\\text{rank} = \\#\\{\\sigma > \\text{tol}\\}"}</M>
                    </div>
                    <p className="text-gray-500 text-sm">
                      Count singular values above a tolerance threshold.
                      Reveals how many truly independent features the design
                      matrix contains.
                    </p>
                  </GlassCard>

                  <GlassCard>
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                      Low-Rank Approximation
                    </p>
                    <div className="mb-2">
                      <M display>{"A_k = U_k\\Sigma_k V_k^T"}</M>
                    </div>
                    <p className="text-gray-500 text-sm">
                      Keep the top-k singular values/vectors. Best rank-k
                      approximation in the Frobenius norm (Eckart-Young
                      theorem).
                    </p>
                  </GlassCard>

                  <GlassCard>
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium mb-2">
                      Ridge Shrinkage
                    </p>
                    <div className="mb-2">
                      <M display>{"\\sigma_i^2 / (\\sigma_i^2 + \\lambda)"}</M>
                    </div>
                    <p className="text-gray-500 text-sm">
                      Ridge regression shrinks each component by a factor that
                      depends on its singular value. Small singular values
                      (noisy directions) are shrunk the most.
                    </p>
                  </GlassCard>
                </div>

                <GlassCard>
                  <SVDShrinkageExplorer />
                </GlassCard>
              </div>
            </section>

            {/* ── Ridge Regression ─────────────── */}
            <section id="ridge" className="py-16">
              <div className="space-y-10">
                <SectionHeader
                  label="Module 6"
                  title="Ridge Regression."
                />

                <div className="space-y-4 text-gray-400 leading-relaxed max-w-3xl">
                  <p>
                    Standard least squares solves{' '}
                    <M>{"\\hat{\\beta} = (X^TX)^{-1}X^Ty"}</M>
                    , but when <M>{"X^TX"}</M> is nearly singular this blows up.
                    Ridge regression adds a penalty term:
                  </p>
                </div>

                <GlassCard>
                  <div className="text-center space-y-2">
                    <p className="text-sm uppercase tracking-widest text-emerald-400 font-medium">
                      Ridge Solution
                    </p>
                    <M display>{"\\hat{\\beta} = (X^TX + \\lambda I)^{-1}X^Ty"}</M>
                    <p className="text-gray-500 text-sm">
                      <M>{"\\lambda I"}</M> makes the matrix invertible even when
                      rank-deficient. The larger <M>{"\\lambda"}</M>, the more we shrink
                      coefficients toward zero.
                    </p>
                  </div>
                </GlassCard>

                {/* Weighted ridge callout */}
                <GlassCard className="border-emerald-400/20">
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-400/10 text-emerald-400 text-sm font-bold shrink-0">
                        !
                      </span>
                      <p className="text-white font-medium">
                        weighted_ridge_solve is THE single most important
                        function in the entire math stack
                      </p>
                    </div>
                    <p className="text-gray-400 text-sm leading-relaxed pl-11">
                      Called by every IRLS iteration. It solves the weighted
                      normal equations where each observation has a different
                      weight based on the current probability estimate.
                    </p>
                    <div className="mt-4">
                      <M display>{"(X^TWX + \\lambda I)\\beta = X^TWz"}</M>
                    </div>
                    <p className="text-gray-500 text-sm text-center">
                      <M>{"W = \\text{diag}(p_i(1 - p_i))"}</M> -- weights from the
                      current logistic predictions. z = working responses.
                    </p>
                  </div>
                </GlassCard>

                {/* Ridge functions table */}
                <GlassCard padding="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/10 text-gray-400 uppercase tracking-wider text-xs">
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">Function</th>
                          <th className="px-3 sm:px-6 py-3 sm:py-4 font-medium">
                            Description
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {RIDGE_FUNCTIONS.map((row) => (
                          <tr
                            key={row.fn}
                            className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
                          >
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-white font-mono text-sm">
                              {row.fn}
                            </td>
                            <td className="px-3 sm:px-6 py-3 sm:py-4 text-gray-300">
                              {row.description}
                              {row.math && (
                                <>
                                  {': '}
                                  <M>{row.math}</M>
                                </>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlassCard>

                <GlassCard>
                  <RidgePathExplorer />
                </GlassCard>
              </div>
            </section>

            {/* ── PrevNextNav ─────────────────── */}
            <section className="pb-24">
              <PrevNextNav
                next={{ label: 'Models', to: '/about/models' }}
              />
            </section>
          </main>
        </div>
      </div>
    </div>
  )
}
