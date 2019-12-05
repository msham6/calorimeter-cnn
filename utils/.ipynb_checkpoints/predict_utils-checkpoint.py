import numpy as np

def prediction(model, first_images, runs_dict, img_size):
    
    num_runs = first_images.shape[0]

    # predictions on first cluster
    first_preds = model.predict(first_images, verbose=1)
    
    # Recenter every run using particle 2 center values and then predict again
    second_images = np.zeros((num_runs, 32, img_size, img_size, 1))
    
    for run in range(num_runs):
        centx2 = runs_dict[str(run)]['particle 2 center'][1]
        centy2 = runs_dict[str(run)]['particle 2 center'][0]
        centx2 = int(centx2); centy2 = int(centy2)
        for x in range(img_size):
            for y in range(img_size):
                if first_images[run, :, x, y, 0].any() > 0.05:
                    second_images[run, :, x-centx2, y-centy2, 0] = first_images[run, :, x, y, 0]
    # predict on second cluster after centering
    second_preds = model.predict(second_images, verbose=1)
    
    # Move back second preds
    second_preds_back = np.zeros((num_runs, 32, 74, 74, 1))
    for run in range(num_runs):
        centx2 = runs_dict[str(run)]['particle 2 center'][1]
        centy2 = runs_dict[str(run)]['particle 2 center'][0]
        centx2 = int(centx2); centy2 = int(centy2)
        for x in range(img_size):
            for y in range(img_size):
                if second_preds[run, :, x, y, 0].any() > 0.05:
                    second_preds_back[run, :, x+centx2, y+centy2, 0] = second_preds[run, :, x, y, 0]
    second_preds_back = second_preds_back[:, :, 0:img_size, 0:img_size, :]
    
    return first_preds, second_preds_back


def share_energy(total_images, first_preds, second_preds, thres1=0.9, thres2=0.1):
    
    num_runs = total_images.shape[0]
    img_size = 48
    
    p1_counts = np.zeros((num_runs, 30, img_size*img_size))
    p2_counts = np.zeros((num_runs, 30, img_size*img_size))
    num_layers = 32

    for example_num in range(num_runs):
        for layer in range(2, num_layers):

            counts_2d = total_images[example_num, layer, :, :, 0]
            counts_2d = np.reshape(counts_2d, (-1,1))

            pred1_ex = first_preds[example_num, layer, :, :, 0]
            pred2_ex_back = second_preds[example_num, layer, :, :, 0]
            preds1 = np.reshape(pred1_ex, (-1,1))
            preds2 = np.reshape(pred2_ex_back, (-1,1))
            combined = pred2_ex_back + pred1_ex
            combined = np.reshape(combined, (-1,1))

            for x in range(img_size*img_size):
                counts = counts_2d[x]
                combined_prob = combined[x]
                if combined_prob > 0.01: 
                    pred1 = preds1[x]/combined_prob
                    pred2 = preds2[x]/combined_prob 
                    if pred1 >= thres1 and pred2 < thres2:
                        p1_counts[example_num, layer-2, x] = counts
                    elif pred2 >= thres1 and pred1 < thres2:
                        p2_counts[example_num, layer-2, x] = counts
                    else:
                        p1_counts[example_num, layer-2, x] += pred1*counts
                        p2_counts[example_num, layer-2, x] += pred2*counts
                        
    return p1_counts, p2_counts

    