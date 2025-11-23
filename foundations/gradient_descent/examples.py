from foundations.gradient_descent.batch_gd import BatchGradientDescent
from foundations.gradient_descent.loss_strategies import MAELoss, MSELoss
from foundations.gradient_descent.mini_batch_gd import MiniBatchGradientDescent
from foundations.gradient_descent.sdg import StochasticGradientDescent
from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector

if __name__ == "__main__":
    # 1. Setup Dummy Data (Linear Relationship: y = 3x1 - 2x2 + 1)
    print("--- Generatng Data ---")
    X = Matrix([
        [1, 1, 1], [1, 2, 1], [1, 1, 2], [1, 3, 2], [1, 2, 3],
        [1, 4, 1], [1, 1, 4], [1, 5, 2], [1, 3, 5], [1, 4, 4]
    ])
    
    # Generate Y based on weights [1, 3, -2] (Bias=1, w1=3, w2=-2)
    target_weights = Vector([1, 3, -2])
    y_data = []
    for i in range(X.num_rows):
        y_val = X.row(i).dot(target_weights)
        y_data.append([y_val])
    y = Matrix(y_data)
    
    print(f"Target Weights: {target_weights}")
    print("-" * 50)

    # 2. Run Batch GD
    print("\n[1] Running Batch Gradient Descent...")
    batch_model = BatchGradientDescent(MSELoss(), learning_rate=0.05, n_epochs=100, verbose=True)
    batch_model.fit(X, y)
    print(f"Batch Result: {batch_model.weights_}")

    # 3. Run SGD
    print("\n[2] Running Stochastic Gradient Descent (SGD)...")
    # Note: Lower learning rate is often needed for SGD to prevent wild oscillation
    sgd_model = StochasticGradientDescent(MSELoss(), learning_rate=0.01, n_epochs=50, verbose=True)
    sgd_model.fit(X, y)
    print(f"SGD Result: {sgd_model.weights_}")

    # 4. Run Mini-Batch GD
    print("\n[3] Running Mini-Batch Gradient Descent...")
    mini_model = MiniBatchGradientDescent(MSELoss(), batch_size=4, learning_rate=0.02, n_epochs=100, verbose=True)
    mini_model.fit(X, y)
    print(f"Mini-Batch Result: {mini_model.weights_}")
    
    print("-" * 50)
    print("Experiment Complete. Notice how all methods converge near [1, 3, -2],")
    print("but they take very different paths to get there!")