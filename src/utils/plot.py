import matplotlib.pyplot as plt


def IOU_plot(train_iou_scores, valid_iou_scores, out_path):
    plt.figure(figsize=(8, 6))

    # Plot IOU scores
    plt.plot(train_iou_scores, label='Training IOU')
    plt.plot(valid_iou_scores, label='Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU Score')
    plt.title('Training and Validation IOU Score')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(out_path)

def Loss_plot(train_losses,valid_losses, out_path ):
    plt.figure(figsize=(8, 6))
    # Plot losses
    # plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(out_path)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()