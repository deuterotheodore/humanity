import numpy as np
import matplotlib.pyplot as plt

# Parameters
optimum = 0.7
optimum_width = 0.05
fitness_benefit = 0.01
mutation_strength = 0.01
population_size = 1000
generations = 4000
initial_value = 0.5

# Initialize population
population = np.full(population_size, initial_value)

# Track statistics over time
mean_history = []
std_history = []
median_history = []

# Run simulation
for gen in range(generations):
    # Calculate fitness for each individual
    # fitness = 1 + benefit * gaussian centered at optimum
    fitness_multiplier = 1 + fitness_benefit * np.exp(-((population - optimum) ** 2) / (2 * optimum_width ** 2))
    
    # Each individual generates a random number weighted by fitness
    random_values = fitness_multiplier * np.random.random(population_size)
    
    # Find median
    median_value = np.median(random_values)
    
    # Select survivors (those above median)
    survivors = population[random_values >= median_value]
    
    # Each survivor produces 2 offspring
    offspring = []
    for parent in survivors:
        for _ in range(2):
            # Add mutation
            child = parent + np.random.normal(0, mutation_strength)
            # Clamp to [0, 1]
            child = np.clip(child, 0, 1)
            offspring.append(child)
    
    # New population (truncate to population_size if needed)
    population = np.array(offspring[:population_size])
    
    # Track statistics
    mean_history.append(np.mean(population))
    std_history.append(np.std(population))
    median_history.append(np.median(population))

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean over time
axes[0, 0].plot(mean_history, label='Mean', linewidth=2)
axes[0, 0].axhline(y=optimum, color='r', linestyle='--', label='Optimum')
axes[0, 0].axhline(y=initial_value, color='gray', linestyle=':', alpha=0.5, label='Initial')
axes[0, 0].set_xlabel('Generation')
axes[0, 0].set_ylabel('Gene Value')
axes[0, 0].set_title('Population Mean Over Time')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Standard deviation over time
axes[0, 1].plot(std_history, linewidth=2, color='orange')
axes[0, 1].axhline(y=mutation_strength, color='purple', linestyle='--', label=f'Mutation SD = {mutation_strength}')
axes[0, 1].axhline(y=optimum_width, color='red', linestyle='--', label=f'Optimum width = {optimum_width}')
axes[0, 1].set_xlabel('Generation')
axes[0, 1].set_ylabel('Standard Deviation')
axes[0, 1].set_title('Population Variance Over Time')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Final distribution histogram
axes[1, 0].hist(population, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=optimum, color='r', linestyle='--', linewidth=2, label='Optimum')
axes[1, 0].axvline(x=np.mean(population), color='blue', linestyle='-', linewidth=2, label=f'Final mean = {np.mean(population):.3f}')
axes[1, 0].set_xlabel('Gene Value')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title(f'Final Distribution (Generation {generations})')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Selection strength metric over time
selection_strength = [(mutation_strength / std_history[i]) ** 2 if std_history[i] > 0 else 0 
                     for i in range(len(std_history))]
axes[1, 1].plot(selection_strength, linewidth=2, color='green')
axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='s = 1')
axes[1, 1].set_xlabel('Generation')
axes[1, 1].set_ylabel('Selection Strength s')
axes[1, 1].set_title(r'Selection Strength: $s = (\sigma_{mut} / \sigma_{obs})^2$')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('/tmp/evolution_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary statistics
print("=" * 60)
print("EVOLUTIONARY SIMULATION RESULTS")
print("=" * 60)
print(f"\nParameters:")
print(f"  Optimum: {optimum}")
print(f"  Optimum width: {optimum_width}")
print(f"  Fitness benefit: {fitness_benefit}")
print(f"  Mutation strength: {mutation_strength}")
print(f"  Initial value: {initial_value}")
print(f"  Generations: {generations}")
print(f"\nFinal Statistics:")
print(f"  Mean: {np.mean(population):.4f}")
print(f"  Median: {np.median(population):.4f}")
print(f"  Std Dev: {np.std(population):.4f}")
print(f"  Distance from optimum: {abs(np.mean(population) - optimum):.4f}")
print(f"\nSelection Strength:")
print(f"  Final s = (σ_mut / σ_obs)² = ({mutation_strength:.3f} / {np.std(population):.3f})² = {(mutation_strength / np.std(population)) ** 2:.2f}")
print(f"\nInterpretation:")
if np.std(population) < mutation_strength:
    print(f"  Strong selection (σ_obs < σ_mut): Distribution tighter than mutation alone")
elif np.std(population) > 2 * mutation_strength:
    print(f"  Weak selection (σ_obs >> σ_mut): Distribution much wider than mutation")
else:
    print(f"  Moderate selection (σ_obs ≈ σ_mut): Mutation-selection balance")
print(f"\n" + "=" * 60)
