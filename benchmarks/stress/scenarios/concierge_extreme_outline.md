# Concierge Extreme (Stress B + D)

## Scenario Goals
- Simulate a high-pressure user offloading a large volume of chores to an AI concierge.
- Emphasize rapid topic switching and complex decision chains.

## Topic Pool (At least 10 parallel threads)
1. Fitness plan
2. Investment portfolio
3. Home renovation
4. Parents' medical checkups
5. Work project A (product launch)
6. Work project B (quarterly report)
7. Learning a new language
8. Dog health
9. Weekend social plans
10. Car maintenance
11. Wedding planning (optional pressure booster)

## Scenario Structure
- **Rounds 1-20**: Jump quickly between topics to establish branches.
- **Rounds 21-50**: Introduce cross-topic requests such as “coordinate fitness and nutrition” or “balance renovation with visiting parents.”
- **Rounds 51-80**: Add unexpected events (flight delays, pet illness) that force the controller to `MERGE` or `SPLIT`.
- **Rounds 81-100**: Multi-topic synthesis requests (e.g., “create a weekly plan that aligns work and family schedules”).

## Key Information Points
- Each topic includes concrete data (time, budget, stakeholders).
- Constraint changes triggered by unexpected events.
- The user expects the AI to remember and reference prior commitments.

## Failure Mode Reminders
- Mixing up threads and reusing information incorrectly.
- Ignoring schedule conflicts created by unexpected events.
- Controller choosing the wrong operation for complex requests, causing a tangled tree structure.
