"""
Prompt set for competitive programming problem generation.

35 distinct prompts with variations in wording and emphasis.
All ask for the same task: Codeforces Div-2 B style problem.
"""

PROBLEM_GENERATION_PROMPTS = [
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


def get_prompt(idx: int) -> str:
    """Get a prompt by index (0-34)."""
    if not 0 <= idx < len(PROBLEM_GENERATION_PROMPTS):
        raise ValueError(f"Prompt index must be 0-{len(PROBLEM_GENERATION_PROMPTS)-1}, got {idx}")
    return PROBLEM_GENERATION_PROMPTS[idx]


def get_all_prompts() -> list[str]:
    """Get all prompts as a list."""
    return PROBLEM_GENERATION_PROMPTS.copy()
