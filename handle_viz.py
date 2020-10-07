total_seeds = 4

import matplotlib.pyplot as plt
import numpy as np
def treat(img):
    nme = img[:]
    img = np.load('figs/'+img)
    print(img.shape, nme)
    if img.ndim==4:
      img = img[0]
    img = img.transpose(1,2,0)
    return img

def rem(img):
  # img[:, :, 2]= np.zeros((img.shape[0], img.shape[1]))
  # img[:, :, 1]= np.zeros((img.shape[0], img.shape[1]))
  return img

def show_images(images, path, cols, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    # assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=(10,20))
    for n, image in enumerate(images):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        # print(sum(image.flatten()))
        a.set_title(titles[n%cols])
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    # plt.show()

    plt.axis('off')
    plt.savefig(path, bbox_inches='tight',pad_inches = 0 )


for use_bullet in range(1):
    for use_different_targets in range(1):
        for use_distractors_in_sender in range(1):
            images = []
            for i in range(6):
                s, r, s_d, r_d=[], [], [], []
                suff = 'i-{}-seed-{}-daware-{}-difftar-{}-bullet-{}'.format(i, 0, use_distractors_in_sender, use_different_targets, use_bullet)
                try:
                  target = treat('target-{}.npy'.format(suff))
                except:
                  print('filenot found', 'target-{}.npy'.format(suff))
                  break
                dist = treat('dist-{}.npy'.format(suff))
                if use_different_targets:
                    target_r = treat('target_r-{}.npy'.format(suff))
                for seed in range(total_seeds):
                    suff = 'i-{}-seed-{}-daware-{}-difftar-{}-bullet-{}'.format(i, seed, use_distractors_in_sender, use_different_targets, use_bullet)

                    heatmap_s = treat('heatmap_s-{}.npy'.format(suff))
                    heatmap_r = treat('heatmap_r-{}.npy'.format(suff))
                    heatmap_r_d = treat('heatmap_r_d-{}.npy'.format(suff))
                    s.append(heatmap_s.copy())
                    r.append(heatmap_r.copy())
                    r_d.append(heatmap_r_d)
                    if use_distractors_in_sender:
                        heatmap_s_d = treat('heatmap_s_d-{}.npy'.format(suff))
                        s_d.append(heatmap_s_d.copy())
                fac = 0.85
                avg_t_s = np.array(target) + fac*rem(np.mean(s, 0))
                images.append(np.array(avg_t_s))

                if use_distractors_in_sender:
                    avg_d_s = dist.copy() + fac*rem(np.mean(s_d, 0))
                    images.append(avg_d_s.copy())
                if use_different_targets:
                    avg_t_r = target_r.copy() + fac*np.mean(r, 0)
                    images.append(avg_t_r.copy())
                else:
                    avg_t_r = target.copy() + fac*np.mean(r, 0)
                    images.append(avg_t_r.copy())
                avg_d_r = dist.copy() + fac*np.mean(r_d, 0)
                images.append(avg_d_r.copy())

            base = 3
            titles = ['Sender Target', 'Receiver Target', 'Receiver Distractor']
            if use_distractors_in_sender:
            	base+=1
            	titles = [titles[0]] + ['Sender Distractor'] + titles[1:]
            if use_different_targets:
              base+=0
            print(base)
            path = 'grid-daware-{}-difftar-{}-bullet-{}'.format(use_distractors_in_sender, use_different_targets, use_bullet)
            try:
              show_images(images, path, base, titles)
            except Exception as e:
              print(e)
              pass
