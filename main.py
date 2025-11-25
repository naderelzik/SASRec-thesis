import psutil
import os
import time
import argparse
import tensorflow as tf
from sampler import *
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--eval_mode', choices=['full', 'sample'], default='sample', help="Evaluation mode: 'full' or 'sample'")
parser.add_argument('--early_stopping_patience', type=int, default=8,
                    help='Number of evaluations to wait before early stopping if no improvement')
parser.add_argument('--sample_mode', type=str, default='normal', choices=['normal', 'random','random_chrono'],
                    help='Choose between normal or random sampling strategy in training')

args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
if args.dataset == 'ml-1m' or args.dataset.startswith('synthetic'):
        dataset = data_partition(args.dataset)
    
elif args.dataset =='foursquare':
        u2i_index, i2u_index = None, None
        folder_path = '/home/zik/SASRec/data/fdata.txt'
        dataset = data_partition_foursquare(folder_path)

elif args.dataset == 'xlong':
    u2i_index, i2u_index = None, None
    folder_path = '/home/zik/SASRec/data/xlong'
    dataset = data_partition_xlong(folder_path)

else:
    raise ValueError("Unsupported dataset")

[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_users = usernum
num_items = itemnum
num_interactions = (
     sum(len(seq) for seq in user_train.values())
      + sum(len(seq) for seq in user_valid.values())
      + sum(len(seq) for seq in user_test.values())
    )

print(f"Dataset = {args.dataset}")
print(f"  # users         = {num_users}")
print(f"  # items         = {num_items}")
print(f"  # interactions  = {num_interactions}")
print(f"  # Sequence length = {args.maxlen}")
num_batch = len(user_train) // args.batch_size
user_train = dataset[0]  # Assuming dataset returned as [user_train, user_valid, user_test, usernum, itemnum]
avg_seq_len = sum(len(seq) for seq in user_train.values()) / len(user_train)
print(f"  # Average sequence length = {avg_seq_len:.2f}")
if args.sample_mode == 'normal':
    sampler_func = sample_function
elif args.sample_mode == 'random':
    sampler_func = sample_random_function
elif args.sample_mode =='random_chrono':
    sampler_func = sample_random_chrono_function
else:
    raise ValueError("Invalid --sampler argument")
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,sampler_func = sampler_func)

model = Model(usernum, itemnum, args)
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
# After graph is built
tf.compat.v1.add_check_numerics_ops()

sess.run(tf.compat.v1.global_variables_initializer())
train_times = []
test_times = []
# Memory Tracking
def track_memory_usage():
    """
    Track and return the peak GPU and CPU memory usage.
    """
    gpu_peak_mem = tf.config.experimental.get_memory_info('GPU:0')['peak'] / (1024 ** 3)  # Convert to GB
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # Convert to GB
    return gpu_peak_mem, cpu_mem
#process = psutil.Process(os.getpid())  # CPU Memory tracking
def format_duration(seconds):
    """Convert seconds to human-readable format with days, hours, minutes, seconds."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:  # Show seconds if it's the only unit
        parts.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")
    
    return ' '.join(parts)
T = 0.0
t0 = time.time()
best_test_metrics = None
patience = args.early_stopping_patience
patience_counter = 0
test_results = []
best_overall_test_ndcg10 = -float('inf')
best_overall_test_ndcg20 = -float('inf')
best_overall_test_metrics = None
best_overall_test_epoch   = 0
best_val_ndcg, best_val_hr = 0.0, 0.0
best_val_ndcg20, best_val_hr20 = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
try:
    #tf.compat.v1.reset_default_graph() 
    for epoch in range(1, args.num_epochs + 1):
        # Reset GPU memory tracker before each epoch
        

        # Training Phase
        train_start_time = time.time()
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})
        # After the inner loop (training) completes:
        print(f"Epoch {epoch} – last‐batch loss: {loss:.4f}, AUC: {auc:.4f}")
        if np.isnan(loss):
                print(f"\nCRITICAL: NaN loss detected at epoch {epoch}, batch {step}")
                break
        train_epoch_time = time.time() - train_start_time
        train_times.append(train_epoch_time)

        # Testing Phase (every 20 epochs)
        if epoch % 20 == 0:
            print('Evaluating...')
            test_start_time = time.time()
            t1 = time.time() - t0
            T += t1
            t_test = evaluate(model, dataset, args, sess)
            
            t_valid = evaluate_valid(model, dataset, args, sess)
            test_epoch_time = (time.time() - test_start_time) #Since this is for 20 epochs only 
            test_times.append(test_epoch_time)

            print('')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, Recall@10: %.4f, NDCG@20: %.4f, Recall@20: %.4f), '
            'test (NDCG@10: %.4f, Recall@10: %.4f, NDCG@20: %.4f, Recall@20: %.4f)'
            % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))


            if t_valid[0] > best_val_ndcg or t_valid[2] > best_val_ndcg20:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_ndcg20 = max(t_valid[2],best_val_ndcg20)
                # folder = args.dataset + '_' + args.train_dir
                # saver.save(sess, f"{folder}/best_model.ckpt")

                best_test_ndcg = t_test[0]
                best_test_hr = t_test[1]
                best_epoch = epoch
                best_test_metrics = t_test
                patience_counter = 0  # reset counter
            if t_test[0] > best_overall_test_ndcg10 or \
                t_test[2] > best_overall_test_ndcg20:
                    best_overall_test_ndcg10 = max(best_overall_test_ndcg10, t_test[0])
                    best_overall_test_ndcg20 = max(best_overall_test_ndcg20, t_test[2])
                    best_overall_test_metrics = t_test
                    best_overall_test_epoch   = epoch
                    
            else:
                patience_counter += 1
                print(f"  --> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break
    # if folder is not None:
    #     saver.restore(sess, f"{folder}/best_model.ckpt")
    final_ndcg10, final_hr10, final_ndcg20, final_hr20 = best_test_metrics
    print(
    f"Test‐driven best at epoch {best_overall_test_epoch}: "
    f"NDCG@10={best_overall_test_metrics[0]:.4f}, "
    f"Recall@10={best_overall_test_metrics[1]:.4f}, "
    f"NDCG@20={best_overall_test_metrics[2]:.4f}, "
    f"Recall@20={best_overall_test_metrics[3]:.4f}"
)        
    # Calculate and print averages
    total_train_time = sum(train_times)
    total_test_time = sum(test_times)
    average_train_time = total_train_time / args.num_epochs if args.num_epochs > 0 else 0
    average_test_time = total_test_time / args.num_epochs if args.num_epochs > 0 else 0

    
    print(f"Total time used: {format_duration(total_train_time + total_test_time)}")
    print(f"Training time: {format_duration(total_train_time)} (avg: {format_duration(average_train_time)} per epoch)")
    print(f"Testing time: {format_duration(total_test_time)} (avg: {format_duration(average_test_time)} per epoch)")
    print(f"Average Train Time per Epoch: {average_train_time:.2f} seconds")
    print(f"Average Test Time per Epoch: {average_test_time:.2f} seconds")

    #Memory usage 
    gpu_peak_mem, cpu_mem = track_memory_usage()
    print(f"\n--- Memory Usage Summary ---")
    print(f"Peak GPU Memory Usage: {gpu_peak_mem:.2f} GB")
    print(f"CPU Memory Usage: {cpu_mem:.2f} GB")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    sampler.close()
    sess.close()
