import matplotlib.pyplot as plt


def visualize_dating_data(data_matrix, labels):
    """
    Create scatter plots to visualize the dating data.
    
    We'll create multiple views to understand the relationships between features.
    This is exactly what the book does with Matplotlib.
    """
    # Convert to separate lists for plotting
    fig = plt.figure(figsize=(15, 5))
    
    # Separate data by class for coloring
    class_1 = [data_matrix[i] for i in range(len(data_matrix)) if labels[i] == 1]
    class_2 = [data_matrix[i] for i in range(len(data_matrix)) if labels[i] == 2]
    class_3 = [data_matrix[i] for i in range(len(data_matrix)) if labels[i] == 3]
    
    # Plot 1: Frequent Flyer Miles vs Percentage Gaming
    plt.subplot(1, 3, 1)
    if class_1:
        plt.scatter([x[0] for x in class_1], [x[1] for x in class_1], 
                   c='red', marker='o', s=50, alpha=0.6, label='Didn\'t Like')
    if class_2:
        plt.scatter([x[0] for x in class_2], [x[1] for x in class_2], 
                   c='green', marker='^', s=50, alpha=0.6, label='Liked in Small Doses')
    if class_3:
        plt.scatter([x[0] for x in class_3], [x[1] for x in class_3], 
                   c='blue', marker='s', s=50, alpha=0.6, label='Liked in Large Doses')
    plt.xlabel('Frequent Flyer Miles per Year')
    plt.ylabel('Percentage Time Gaming')
    plt.legend()
    plt.title('Dating Data: Miles vs Gaming')
    
    # Plot 2: Gaming vs Ice Cream  
    plt.subplot(1, 3, 2)
    if class_1:
        plt.scatter([x[1] for x in class_1], [x[2] for x in class_1], 
                   c='red', marker='o', s=50, alpha=0.6, label='Didn\'t Like')
    if class_2:
        plt.scatter([x[1] for x in class_2], [x[2] for x in class_2], 
                   c='green', marker='^', s=50, alpha=0.6, label='Liked in Small Doses')
    if class_3:
        plt.scatter([x[1] for x in class_3], [x[2] for x in class_3], 
                   c='blue', marker='s', s=50, alpha=0.6, label='Liked in Large Doses')
    plt.xlabel('Percentage Time Gaming')
    plt.ylabel('Liters Ice Cream per Week')
    plt.legend()
    plt.title('Dating Data: Gaming vs Ice Cream')
    
    # Plot 3: Miles vs Ice Cream
    plt.subplot(1, 3, 3)
    if class_1:
        plt.scatter([x[0] for x in class_1], [x[2] for x in class_1], 
                   c='red', marker='o', s=50, alpha=0.6, label='Didn\'t Like')
    if class_2:
        plt.scatter([x[0] for x in class_2], [x[2] for x in class_2], 
                   c='green', marker='^', s=50, alpha=0.6, label='Liked in Small Doses')
    if class_3:
        plt.scatter([x[0] for x in class_3], [x[2] for x in class_3], 
                   c='blue', marker='s', s=50, alpha=0.6, label='Liked in Large Doses')
    plt.xlabel('Frequent Flyer Miles per Year')
    plt.ylabel('Liters Ice Cream per Week')
    plt.legend()
    plt.title('Dating Data: Miles vs Ice Cream')
    
    plt.tight_layout()
    plt.show()