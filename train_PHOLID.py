import json
import scoring
import subprocess
from ssl_sampler import *
from model import *
from data_load import *
import torch.nn as nn
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_


def validation(valid_txt, model, model_name, device, kaldi, log_dir, num_lang):
    valid_set = RawFeatures(valid_txt)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = 0
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            # Forward pass\
            outputs, _ = model(utt, seq_len)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            if step == 0:
                scores = outputs
            else:
                scores = torch.cat((scores, outputs), dim=0)
    acc = correct / total
    print('Current Acc.: {:.4f} %'.format(100 * acc))
    scores = scores.squeeze().cpu().numpy()
    print(scores.shape)
    trial_txt = log_dir + '/trial_{}.txt'.format(model_name)
    score_txt = log_dir + '/score_{}.txt'.format(model_name)
    output_txt = log_dir + '/output_{}.txt'.format(model_name)
    scoring.get_trials(valid_txt, num_lang, trial_txt)
    scoring.get_score(valid_txt, scores, num_lang, score_txt)
    eer_txt = trial_txt.replace('trial', 'eer')
    subprocess.call(f"{kaldi}/egs/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
    cavg = scoring.compute_cavg(trial_txt, score_txt)
    print("Cavg:{}".format(cavg))
    with open(output_txt, 'w') as f:
        f.write("ACC:{} Cavg:{}".format(acc, cavg))
    return cavg


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--json', type=str, default='xsa_config.json')
    args = parser.parse_args()
    with open(args.json, 'r') as json_obj:
        config_proj = json.load(json_obj)
    seed = config_proj["optim_config"]["seed"]
    if seed == -1:
        pass
    else:
        print("Random seed is {}".format(seed))
        setup_seed(seed)
    device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
                          if torch.cuda.is_available() else 'cpu')
    feat_dim = config_proj["model_config"]["d_k"]
    n_heads = config_proj["model_config"]["n_heads"]
    model = PHOLID(input_dim=config_proj["model_config"]["feat_dim"],
                   feat_dim=config_proj["model_config"]["d_k"],
                   d_k=config_proj["model_config"]["d_k"],
                   d_v=config_proj["model_config"]["d_k"],
                   d_ff=config_proj["model_config"]["d_ff"],
                   n_heads=config_proj["model_config"]["n_heads"],
                   dropout=0.1,
                   n_lang=config_proj["model_config"]["n_language"],
                   max_seq_len=10000)

    model.to(device)
    model_name = config_proj["model_name"]
    print("model name: {}".format(model_name))
    log_dir = config_proj["Input"]["userroot"] + config_proj["Input"]["log"]
    kaldi_root = config_proj["Input"]["userroot"] + config_proj["kaldi"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["train"]
    train_set = RawFeatures(train_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=config_proj["optim_config"]["batch"],
                            pin_memory=True,
                            num_workers=config_proj["optim_config"]["num_work"],
                            shuffle=True,
                            collate_fn=collate_fn_atten)

    if config_proj["Input"]["valid"] != "none":
        print("Validation is True")
        valid_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["valid"]
    else:
        valid_txt = None
    if config_proj["Input"]["test"] != "none":
        print("Test is True")
        test_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["test"]
    else:
        test_txt = None

    loss_func_lid = nn.CrossEntropyLoss().to(device)
    if config_proj["optim_config"]["phn_ssl"] == "seg":
        num_nega_samples = config_proj["optim_config"]["nega_frames"]
        print("Compute phoneme SSL over segments with {} negative samples".format(num_nega_samples))
        loss_func_phn = Phoneme_SSL_loss(num_frames=20, num_sample=num_nega_samples).to(device)
    elif config_proj["optim_config"]["phn_ssl"] == "utt":
        num_nega_samples = config_proj["optim_config"]["nega_frames"]
        print("Compute phoneme SSL over utterances with {} negative samples".format(num_nega_samples))
        loss_func_phn = Phoneme_SSL_utt_loss(num_sample=num_nega_samples).to(device)
    else:
        raise ValueError("The phn_ssl should be either 'seg' or 'utt' ")
    total_step = len(train_data)
    total_epochs = config_proj["optim_config"]["epochs"]
    valid_epochs = config_proj["optim_config"]["valid_epochs"]
    weight_lid = config_proj["optim_config"]["weight_lid"]
    weight_ssl = config_proj["optim_config"]["weight_ssl"]
    print("weights: LID {} SSL {}".format(weight_lid, weight_ssl))
    optimizer = torch.optim.Adam(model.parameters(), lr=config_proj["optim_config"]["learning_rate"])
    SSL_epochs = config_proj["optim_config"]["SSL_epochs"]
    SSL_steps = SSL_epochs * total_step
    if config_proj["optim_config"]["warmup_step"] == -1:
        warmup = total_step * 3
    else:
        warmup = config_proj["optim_config"]["warmup_step"]
    if config_proj["optim_config"]["warmup_step"] == -1:
        warmup = total_step * 3
    else:
        warmup = config_proj["optim_config"]["warmup_step"]
    warm_up_with_cosine_lr = lambda step: 1 if step <= SSL_steps else (
        (step - SSL_steps) / warmup if step < SSL_steps + warmup else 0.5 * (
                math.cos((step - SSL_steps - warmup) / (total_epochs * total_step - SSL_steps - warmup) * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    for epoch in tqdm(range(total_epochs)):
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            mean_mask_, weight_mean = mean_mask(seq_len, len(seq_len), dim=feat_dim * n_heads)
            std_mask_, weight_unbaised = std_mask(seq_len, len(seq_len), dim=feat_dim * n_heads)
            mean_mask_ = mean_mask_.to(device)
            weight_mean = weight_mean.to(device)
            std_mask_ = std_mask_.to(device=device)
            weight_unbaised = weight_unbaised.to(device=device)
            labels = labels.to(device=device)
            # Forward pass
            outputs, phonemes = model(utt_, seq_len, mean_mask_, weight_mean, std_mask_, weight_unbaised,
                                    atten_mask=atten_mask)
            # Backward and optimize
            if epoch < SSL_epochs:
                loss_phn = loss_func_phn(phonemes, seq_len)
                loss_lid = loss_func_lid(outputs, labels)
                loss = loss_phn
            else:
                loss_lid = loss_func_lid(outputs, labels)
                loss_phn = loss_func_phn(phonemes, seq_len)
                loss = weight_lid*loss_lid+weight_ssl*loss_phn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} LID: {:.4f} PHN: {:.4f}".
                      format(epoch + 1, total_epochs, step + 1, total_step, loss.item(),
                             loss_lid.item(), loss_phn.item()))
        torch.save(model.state_dict(), '{}_epoch_{}.ckpt'.format(model_name, epoch))
        if epoch >= total_epochs - valid_epochs - 1:
            if valid_txt is not None:
                validation(valid_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                           num_lang=config_proj["model_config"]["n_language"])
            if test_txt is not None:
                validation(test_txt, model, model_name, device, kaldi=kaldi_root, log_dir=log_dir,
                           num_lang=config_proj["model_config"]["n_language"])


if __name__ == "__main__":
    main()
