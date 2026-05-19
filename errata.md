_Last updated 19 May 2026_

# Errata
On page 19 and 20, all 3 occurrences of $`\min(\delta_t^2, \epsilon_\textup{err}^2)`$ should instead be $`\max(\delta_t^2, \epsilon_\textup{err}^2)`$, as the error clipping should cause the error-clipped replacement term to not be below $`\epsilon_\textup{err}^2`$. Other than this mistake, analysis and implementation of this does not change.

# Relevant developments
None so far.

