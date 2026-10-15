# MACHINE-CHALLENGE-7-COMP3
This repository holds the files for the Machine Challenge 7, it contains the API, .html, and .py. 

10-15-26
Refactor codebase, optimize UI/solver performance, and split static assets

1. Extract inline CSS and JS from mc7_interactive.html into style.css and script.js.
2. Replace directory-wide StaticFiles mount and wildcard CORS in mc7_api.py with explicit asset routes and localhost-restricted origins.
3. Remove radial-gradient background hues and backdrop-filter blur effects in style.css to eliminate GPU lag and improve presentation readability.
4. Consolidate duplicate CSS rules into 6 structured sections.
5. Optimize gauss_jordan_steps() and solve_with_inverse() in mc7_matrix_inverse.py by pre-allocating the identity matrix, skipping zero multipliers in matmul(), and computing det(A) directly from pivots/swaps.
6. Eliminate double JSON serialization in POST /api/solve.
7. Improve frontend responsiveness in script.js by collapsing heavy step drawers by default, deferring textarea sync, and minimizing DOM attribute mutations.
8. Fix fraction regex to allow signed denominators (e.g., 3/-4), fix minus-sign formatting in scientific notation, and validate CLI --random arguments.
9. Replace assigned lambdas with named functions and add function/output documentation across Python and JS files.
