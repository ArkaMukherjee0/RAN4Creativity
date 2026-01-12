"""
Prompt sets for competitive programming problem generation.

Two prompt lists:
- PROBLEM_GENERATION_PROMPTS_CF: 35 prompts for Codeforces Div-2 B style problems
- PROBLEM_GENERATION_PROMPTS_LC: 20 prompts for LeetCode Medium-Hard problems
"""

# ============================================================================
# Codeforces Prompts (Div-2 B difficulty)
# ============================================================================

PROBLEM_GENERATION_PROMPTS_CF = [
    # Prompt 0: Base template
    """Generate a competitive programming problem suitable for Codeforces Div-2 B.
Include:
• problem statement
• input format
• output format
• constraints
• short solution outline.""",

    # Prompt 1: Slightly different phrasing
    """Create a competitive programming problem at Codeforces Division 2 difficulty, problem B level.
The problem should include:
• A clear problem statement
• Input format specification
• Output format specification
• Constraint details
• Brief solution approach.""",

    # Prompt 2: More directive tone
    """Design a Codeforces Div-2 B difficulty problem.
Provide:
• Problem description
• Input specification
• Output specification
• Constraints
• Solution outline.""",

    # Prompt 3: Emphasize completeness
    """Write a complete competitive programming problem for Codeforces Division 2, problem B.
Must contain:
• Full problem statement
• Input format
• Output format
• All constraints
• Short solution strategy.""",

    # Prompt 4: Alternative structure
    """Compose a Codeforces-style programming problem, Div-2 B difficulty.
Include these sections:
• Statement of the problem
• Input requirements
• Output requirements
• Constraint bounds
• Brief solving approach.""",

    # Prompt 5: Focus on completeness
    """Generate a programming contest problem suitable for Codeforces Division 2, position B.
Ensure it has:
• Problem statement
• Input format
• Output format
• Constraints
• Solution outline.""",

    # Prompt 6: Slightly formal
    """Construct a competitive programming problem matching Codeforces Div-2 B specifications.
Requirements:
• Problem statement
• Input format
• Output format
• Constraints
• Solution sketch.""",

    # Prompt 7: Emphasize structure
    """Create a structured competitive programming problem for Codeforces Division 2, level B.
Components needed:
• Problem description
• Input specification
• Output specification
• Constraint details
• Solution approach.""",

    # Prompt 8: Direct and concise
    """Write a Codeforces Div-2 B problem with:
• Problem statement
• Input format
• Output format
• Constraints
• Solution outline.""",

    # Prompt 9: Slightly varied emphasis
    """Develop a competitive programming problem at Codeforces Div-2 B difficulty.
Include:
• Detailed problem statement
• Input format
• Output format
• All constraints
• Brief solution method.""",

    # Prompt 10: Algorithm-focused
    """Create a Codeforces Div-2 B level problem with a clear algorithmic solution.
Provide:
• Problem statement
• Input/output formats
• Constraints
• Solution strategy.""",

    # Prompt 11: Story-based emphasis
    """Write a competitive programming problem (Codeforces Div-2 B) with an engaging narrative.
Include:
• Story-based problem statement
• Input format
• Output format
• Constraints
• Solution hint.""",

    # Prompt 12: Clarity emphasis
    """Produce a clear and well-structured Codeforces Div-2 B problem.
Requirements:
• Unambiguous problem statement
• Input specification
• Output specification
• Constraint bounds
• Solution overview.""",

    # Prompt 13: Testcase emphasis
    """Generate a Codeforces Div-2 B problem with sample test cases.
Must have:
• Problem statement
• Input/output formats
• Constraints
• Sample inputs and outputs
• Solution approach.""",

    # Prompt 14: Difficulty calibration
    """Create a medium-difficulty competitive programming problem (Codeforces Div-2 B standard).
Include:
• Problem description
• Input format
• Output format
• Constraints
• Brief solution.""",

    # Prompt 15: Implementation-focused
    """Design a Codeforces Div-2 B problem that tests implementation skills.
Provide:
• Problem statement
• Input specification
• Output specification
• Constraints
• Implementation hints.""",

    # Prompt 16: Concise variant
    """Codeforces Div-2 B problem needed.
Include: problem statement, I/O formats, constraints, solution outline.""",

    # Prompt 17: Educational tone
    """Formulate an educational competitive programming problem at Codeforces Div-2 B level.
Contents:
• Problem statement
• Input format
• Output format
• Constraints
• Learning objectives and solution.""",

    # Prompt 18: Performance emphasis
    """Write a Codeforces Div-2 B problem focusing on time complexity.
Include:
• Problem statement
• Input/output formats
• Constraints with N bounds
• Expected time complexity
• Solution approach.""",

    # Prompt 19: Data structure focus
    """Create a Codeforces Div-2 B problem involving data structures.
Provide:
• Problem statement
• Input format
• Output format
• Constraints
• Data structure hints.""",

    # Prompt 20: Pattern recognition
    """Generate a Codeforces Div-2 B problem based on pattern recognition.
Include:
• Problem description
• Input specification
• Output specification
• Constraints
• Pattern hints.""",

    # Prompt 21: Mathematical emphasis
    """Design a math-oriented Codeforces Div-2 B problem.
Requirements:
• Problem statement
• Input format
• Output format
• Mathematical constraints
• Solution strategy.""",

    # Prompt 22: String manipulation
    """Create a Codeforces Div-2 B problem involving string operations.
Must include:
• Problem statement
• Input/output formats
• String constraints
• Solution outline.""",

    # Prompt 23: Array operations
    """Write a Codeforces Div-2 B problem centered on array manipulation.
Provide:
• Problem description
• Input format
• Output format
• Array constraints
• Solution approach.""",

    # Prompt 24: Greedy algorithm
    """Produce a Codeforces Div-2 B problem solvable with a greedy approach.
Include:
• Problem statement
• Input specification
• Output specification
• Constraints
• Greedy strategy hint.""",

    # Prompt 25: Two-pointer technique
    """Generate a Codeforces Div-2 B problem that can use two-pointer technique.
Contents:
• Problem statement
• Input/output formats
• Constraints
• Solution hint.""",

    # Prompt 26: Binary search
    """Create a Codeforces Div-2 B problem where binary search is applicable.
Include:
• Problem description
• Input format
• Output format
• Constraints
• Search space hint.""",

    # Prompt 27: Graph basics
    """Design a Codeforces Div-2 B problem with basic graph concepts.
Provide:
• Problem statement
• Input specification (nodes/edges)
• Output specification
• Graph constraints
• Solution approach.""",

    # Prompt 28: Sorting-based
    """Write a Codeforces Div-2 B problem where sorting is key.
Include:
• Problem statement
• Input format
• Output format
• Constraints
• Sorting strategy.""",

    # Prompt 29: Prefix sums
    """Generate a Codeforces Div-2 B problem utilizing prefix sum technique.
Must have:
• Problem description
• Input/output formats
• Constraints
• Prefix sum hint.""",

    # Prompt 30: Modular arithmetic
    """Create a Codeforces Div-2 B problem involving modular arithmetic.
Requirements:
• Problem statement
• Input format
• Output format (often mod 10^9+7)
• Constraints
• Solution outline.""",

    # Prompt 31: Combinatorics
    """Produce a Codeforces Div-2 B problem with combinatorial elements.
Include:
• Problem statement
• Input specification
• Output specification
• Constraints
• Counting strategy.""",

    # Prompt 32: Observation-based
    """Write a Codeforces Div-2 B problem requiring a key observation.
Provide:
• Problem description
• Input/output formats
• Constraints
• Observation hint.""",

    # Prompt 33: Simulation
    """Design a Codeforces Div-2 B problem requiring simulation.
Contents:
• Problem statement
• Input format
• Output format
• Constraints
• Simulation approach.""",

    # Prompt 34: Constructive algorithm
    """Generate a Codeforces Div-2 B constructive problem.
Include:
• Problem statement
• Input specification
• Output specification (construct valid solution)
• Constraints
• Construction strategy.""",
]


# ============================================================================
# LeetCode Prompts (Medium to Hard difficulty)
# ============================================================================

PROBLEM_GENERATION_PROMPTS_LC = [
    # Prompt 0: Base template
    """Generate a LeetCode-style algorithm problem at Medium or Hard difficulty (no Easy problems).
Follow the standard LeetCode problem format:
• Problem title
• Problem description with clear narrative
• Function signature to implement
• Constraints section
• Example inputs and outputs with explanations
• Brief solution approach.""",

    # Prompt 1: Emphasize difficulty
    """Create a LeetCode Medium-Hard difficulty algorithm problem.
Must follow LeetCode's official problem structure:
• Title
• Description
• Function signature (e.g., def solve(nums: List[int]) -> int)
• Constraints
• Examples with input/output/explanation
• Solution hint.""",

    # Prompt 2: Array focus
    """Write a LeetCode Medium or Hard problem involving array manipulation.
Use LeetCode's standard format:
• Problem title
• Detailed problem statement
• Function signature to implement
• Constraints (array size, element bounds)
• 2-3 examples with explanations
• Solution approach.""",

    # Prompt 3: String manipulation
    """Design a LeetCode Medium-Hard string problem.
Follow LeetCode problem writing structure:
• Title
• Problem description
• Function signature
• String constraints (length, character set)
• Example cases with explanations
• Brief solution strategy.""",

    # Prompt 4: Dynamic programming
    """Generate a LeetCode Medium or Hard problem solvable with dynamic programming.
Adhere to LeetCode format:
• Problem title
• Clear problem statement
• Function signature
• Constraints
• Examples with step-by-step explanation
• DP approach hint.""",

    # Prompt 5: Graph problem
    """Create a LeetCode Medium-Hard graph algorithm problem.
Use standard LeetCode structure:
• Title
• Problem description with graph context
• Function signature (adjacency list or edge list input)
• Constraints (nodes, edges)
• Examples with visual explanation if needed
• Solution outline.""",

    # Prompt 6: Tree problem
    """Write a LeetCode Medium or Hard binary tree problem.
Follow LeetCode's format:
• Problem title
• Tree-based problem statement
• Function signature with TreeNode parameter
• Constraints (number of nodes, value range)
• Examples showing tree structure
• Solution approach.""",

    # Prompt 7: Two pointers
    """Design a LeetCode Medium-Hard problem solvable with two-pointer technique.
LeetCode format required:
• Title
• Problem description
• Function signature
• Constraints
• Examples with explanations
• Two-pointer hint.""",

    # Prompt 8: Sliding window
    """Generate a LeetCode Medium or Hard sliding window problem.
Must follow LeetCode structure:
• Problem title
• Detailed description
• Function signature to implement
• Constraints
• Example inputs/outputs with explanations
• Window approach hint.""",

    # Prompt 9: Binary search
    """Create a LeetCode Medium-Hard problem where binary search is the optimal approach.
Use LeetCode's problem format:
• Title
• Problem statement
• Function signature
• Constraints (sorted input or search space)
• Examples with explanations
• Search space hint.""",

    # Prompt 10: Hash map/set
    """Write a LeetCode Medium or Hard problem requiring hash-based data structures.
Follow LeetCode format:
• Problem title
• Problem description
• Function signature
• Constraints
• Examples with explanations
• Hash strategy hint.""",

    # Prompt 11: Stack/Queue
    """Design a LeetCode Medium-Hard problem using stack or queue.
LeetCode structure required:
• Title
• Problem statement
• Function signature
• Constraints
• Examples with step-by-step explanation
• Data structure hint.""",

    # Prompt 12: Heap/Priority queue
    """Generate a LeetCode Medium or Hard problem solvable with heaps.
Must use LeetCode format:
• Problem title
• Detailed description
• Function signature
• Constraints
• Examples with explanations
• Heap approach hint.""",

    # Prompt 13: Backtracking
    """Create a LeetCode Medium-Hard backtracking problem.
Follow LeetCode's standard structure:
• Title
• Problem description
• Function signature (often returns list of solutions)
• Constraints
• Examples showing all valid outputs
• Backtracking strategy.""",

    # Prompt 14: Greedy algorithm
    """Write a LeetCode Medium or Hard problem with a greedy solution.
Use LeetCode format:
• Problem title
• Problem statement
• Function signature
• Constraints
• Examples with explanations
• Greedy insight hint.""",

    # Prompt 15: Bit manipulation
    """Design a LeetCode Medium-Hard bit manipulation problem.
LeetCode structure:
• Title
• Problem description
• Function signature
• Constraints (integer bounds)
• Examples with binary explanations
• Bit operation hint.""",

    # Prompt 16: Linked list
    """Generate a LeetCode Medium or Hard linked list problem.
Follow LeetCode format:
• Problem title
• Detailed description
• Function signature with ListNode parameter
• Constraints (list length, values)
• Examples showing list transformations
• Solution approach.""",

    # Prompt 17: Matrix/2D array
    """Create a LeetCode Medium-Hard matrix problem.
Must use LeetCode structure:
• Problem title
• Problem statement with grid context
• Function signature (2D array input)
• Constraints (rows, columns, values)
• Examples with grid visualization
• Solution strategy.""",

    # Prompt 18: Union-Find
    """Write a LeetCode Medium or Hard problem solvable with Union-Find.
Follow LeetCode format:
• Title
• Problem description (connectivity/grouping theme)
• Function signature
• Constraints
• Examples with explanations
• Union-Find hint.""",

    # Prompt 19: Trie
    """Design a LeetCode Medium-Hard problem involving Trie data structure.
LeetCode format required:
• Problem title
• Problem statement (prefix/word search theme)
• Class or function signature
• Constraints (word count, length)
• Examples with explanations
• Trie approach hint.""",
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_prompt(idx: int, platform: str = "codeforces") -> str:
    """
    Get a prompt by index.

    Args:
        idx: Prompt index
        platform: "codeforces" or "leetcode"

    Returns:
        The prompt string
    """
    prompts = PROBLEM_GENERATION_PROMPTS_CF if platform == "codeforces" else PROBLEM_GENERATION_PROMPTS_LC
    if not 0 <= idx < len(prompts):
        raise ValueError(f"Prompt index must be 0-{len(prompts)-1}, got {idx}")
    return prompts[idx]


def get_all_prompts(platform: str = "codeforces") -> list[str]:
    """
    Get all prompts for a platform.

    Args:
        platform: "codeforces" or "leetcode"

    Returns:
        List of prompt strings
    """
    if platform == "codeforces":
        return PROBLEM_GENERATION_PROMPTS_CF.copy()
    elif platform == "leetcode":
        return PROBLEM_GENERATION_PROMPTS_LC.copy()
    else:
        raise ValueError(f"Unknown platform: {platform}. Use 'codeforces' or 'leetcode'")
