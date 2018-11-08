import argparse
import numpy as np
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def print_distortion_distance(cluster_prototypes, points_by_label, k):
    distances = np.zeros((k,))

    for k_i in range(k):
        if (points_by_label[k_i] is not None):
            distances[k_i] += np.linalg.norm(points_by_label[k_i] - cluster_prototypes[k_i], axis=1).sum()
        else:
            distances[k_i] = -1

    print('Distortion Distances:')
    print(distances)


def k_means_clustering(image_vectors, k, num_iterations):
    # Create corresponding label array (Initialize with Label: -1)
    labels = np.full((image_vectors.shape[0],), -1)

    # Assign Initial Cluster Prototypes
    cluster_prototypes = np.random.rand(k, 3)

    # Iteration Loop
    for i in range(num_iterations):
        print('Iteration: ' + str(i + 1))
        points_by_label = [None for k_i in range(k)]

        # Label them via closest point
        for rgb_i, rgb in enumerate(image_vectors):
            # [rgb, rgb, rgb, rgb, ...]
            rgb_row = np.repeat(rgb, k).reshape(3, k).T

            # Find the Closest Label via L2 Norm
            closest_label = np.argmin(np.linalg.norm(rgb_row - cluster_prototypes, axis=1))
            labels[rgb_i] = closest_label

            if (points_by_label[closest_label] is None):
                points_by_label[closest_label] = []

            points_by_label[closest_label].append(rgb)

        # Optimize Cluster Prototypes (Center of Mass of Cluster)
        for k_i in range(k):
            if (points_by_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_by_label[k_i]).sum(axis=0) / len(points_by_label[k_i])
                cluster_prototypes[k_i] = new_cluster_prototype

        # Find Current Distortion Distances
        print_distortion_distance(cluster_prototypes, points_by_label, k)

    return (labels, cluster_prototypes)

# NOTE: UNUSED
def assign_image_color_by_label(labels, cluster_prototypes):
    output = np.zeros(labels.shape + (3,))

    for label_i, label in enumerate(labels):
        output[label_i] = cluster_prototypes[label]

    return output


def plot_image_colors_by_color(name, image_vectors):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb in image_vectors:
        ax.scatter(rgb[0], rgb[1], rgb[2], c=rgb, marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')


def plot_image_colors_by_label(name, image_vectors, labels, cluster_prototypes):
    fig = plt.figure()
    ax = Axes3D(fig)

    for rgb_i, rgb in enumerate(image_vectors):
        ax.scatter(rgb[0], rgb[1], rgb[2], c=cluster_prototypes[labels[rgb_i]], marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    fig.savefig(name + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='image_compression', description='k-means Image Compressor', add_help=False)
    parser.add_argument('image_name', type=str, help='Image Filename')
    parser.add_argument('-k', type=int, dest='k', help='Number of Clusters', default=10)
    parser.add_argument('-i', '--iterations', type=int, dest='iterations', help='Number of Iterations', default=20)
    parser.add_argument('--save-scatter', dest='scatter', action='store_true')
    parser.set_defaults(scatter=False)

    args = parser.parse_args()
    params = vars(args)

    image = io.imread(params['image_name'])[:,:,:3]  # Always read it as RGB (ignoring the Alpha)
    image = image / 255  # Scale it to [0,1]

    image_dimensions = image.shape
    # Get Image Name without the extension
    image_tokens = params['image_name'].split('.')
    image_name = '.'.join(params['image_name'].split('.')[:-1]) if len(image_tokens) > 1 else params['image_name']

    # -1 infers dimensions from the length of the matrix, while keeping the last dimension a 3-tuple
    image_vectors = image.reshape(-1, image.shape[-1])

    if (params['scatter']):
        print('Creating Initial Scatter Plot! Might take a while...')
        plot_image_colors_by_color('Initial_Colors_' + image_name, image_vectors)
        print('Scatter Plot Complete')

    labels, color_centroids = k_means_clustering(image_vectors, k=params['k'], num_iterations=params['iterations'])

    output_image = np.zeros(image_vectors.shape)
    for i in range(output_image.shape[0]):
        output_image[i] = color_centroids[labels[i]]

    output_image = output_image.reshape(image_dimensions)

    print('Saving Compressed Image...')
    io.imsave(image_name + '_compressed_' + str(params['k']) + '.png', output_image);
    print('Image Compression Completed!')

    if (params['scatter']):
        print('Creating Output Scatter Plot! Might take a while...')
        plot_image_colors_by_label(str(params['k']) + '_Cluster_Colors_' + image_name, image_vectors, labels, color_centroids)
        print('Scatter Plot Complete')
