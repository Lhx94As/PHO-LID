import os

def get_trials(utt2lan, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(' ')[0])[-1].strip('.npy') for x in lines]
    lang_list = [int(x.split(' ')[1].strip()) for x in lines]
    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
            target_utt = lang_list[i]
            utt = utt_list[i]
            for target in targets:
                if target == target_utt:
                    f.write("{} {} target\n".format(utt, target))
                else:
                    f.write("{} {} nontarget\n".format(utt, target))

def get_score(utt2lan, scores, num_lang, output):
    with open(utt2lan, 'r') as f:
        lines = f.readlines()
    utt_list = [os.path.split(x.split(' ')[0])[-1].strip('.npy') for x in lines]
    lang_list = [int(x.split(' ')[-1].strip()) for x in lines]
    targets = [i for i in range(num_lang)]
    with open(output, 'w') as f:
        for i in range(len(utt_list)):
            score_utt = scores[i]
            for lang_id in targets:
                str_ = "{} {} {}\n".format(utt_list[i], lang_id, score_utt[lang_id])
                f.write(str_)

def get_langid_dict(trials):
  ''' Get lang2lang_id, utt2lang_id dicts and lang nums, lang_id starts from 0.
      Also return trial list.
  '''
  langs = []
  lines = open(trials, 'r').readlines()
  for line in lines:
    utt, lang, target = line.strip().split()
    langs.append(lang)

  langs = list(set(langs))
  langs.sort()
  lang2lang_id = {}
  for i in range(len(langs)):
    lang2lang_id["{}".format(i)] = i

  utt2lang_id = {}
  trial_list = {}
  for line in lines:
    utt, lang, target = line.strip().split()
    if target == 'target':
      utt2lang_id[utt] = lang2lang_id[lang]
    trial_list[lang + utt] = target

  return lang2lang_id, utt2lang_id, len(langs), trial_list


def process_pair_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Replace both lang names and utt ids with their lang ids,
      for unknown utt, just with -1. Also return the min and max scores.
  '''
  pairs = []
  stats = []
  lines = open(scores, 'r').readlines()
  for line in lines:
    utt, lang, score = line.strip().split()
    if lang + utt in trial_list:
      if utt in utt2lang_id:
        pairs.append([lang2lang_id[lang], utt2lang_id[utt], float(score)])
      else:
        pairs.append([lang2lang_id[lang], -1, float(score)])
      stats.append(float(score))
  return pairs, min(stats), max(stats)


def process_matrix_scores(scores, lang2lang_id, utt2lang_id, lang_num, trial_list):
  ''' Convert matrix scores to pairs as returned by process_pair_scores.
  '''
  lines = open(scores, 'r').readlines()
  langs_order = {} # langs order in the first line of scores
  langs = lines[1].strip().split()
  for i in range(len(langs)):
    langs_order[i] = langs[i]

  pairs = []
  stats = []
  for line in lines[0:]:
    items = line.strip().split()
    utt = items[0]
    sco = items[2:]
    for i in range(len(sco)):
      if langs_order[i] + utt in trial_list:
        if utt in utt2lang_id:
          pairs.append([lang2lang_id[langs_order[i]], utt2lang_id[utt], float(sco[i])])
        else:
          pairs.append([lang2lang_id[langs_order[i]], -1, float(sco[i])])
        stats.append(float(sco[i]))
  return pairs, min(stats), max(stats)


def get_cavg(pairs, lang_num, min_score, max_score, bins = 20, p_target = 0.5):
  ''' Compute Cavg, using several threshhold bins in [min_score, max_score].
  '''
  cavgs = [0.0] * (bins + 1)
  precision = (max_score - min_score) / bins
  for section in range(bins + 1):
    threshold = min_score + section * precision
    # Cavg for each lang: p_target * p_miss + sum(p_nontarget*p_fa)
    target_cavg = [0.0] * lang_num
    for lang in range(lang_num):
      p_miss = 0.0 # prob of missing target pairs
      LTa = 0.0 # num of all target pairs
      LTm = 0.0 # num of missing pairs
      p_fa = [0.0] * lang_num # prob of false alarm, respect to all other langs
      LNa = [0.0] * lang_num # num of all nontarget pairs, respect to all other langs
      LNf = [0.0] * lang_num # num of false alarm pairs, respect to all other langs
      for line in pairs:
        if line[0] == lang:
          if line[1] == lang:
            LTa += 1
            if line[2] < threshold:
              LTm += 1
          else:
            LNa[line[1]] += 1
            if line[2] >= threshold:
              LNf[line[1]] += 1
      if LTa != 0.0:
        p_miss = LTm / LTa
      for i in range(lang_num):
        if LNa[i] != 0.0:
          p_fa[i] = LNf[i] / LNa[i]
      p_nontarget = (1 - p_target) / (lang_num - 1)
      target_cavg[lang] = p_target * p_miss + p_nontarget*sum(p_fa)
    cavgs[section] = sum(target_cavg) / lang_num
  return cavgs, min(cavgs)


def compute_cavg(trial_txt, score_txt, p_target=0.5):
    '''
    :param trail: trial file
    :param score: score file
    :param p_target: default 0.5
    :return: Cavg (average cost)
    '''
    lang2lang_id, utt2lang_id, lang_num, trial_list = get_langid_dict(trial_txt)
    pairs, min_score, max_score = process_pair_scores(score_txt, lang2lang_id, utt2lang_id, lang_num, trial_list)
    threshhold_bins = 20
    p_target = p_target
    cavgs, min_cavg = get_cavg(pairs, lang_num, min_score, max_score, threshhold_bins, p_target)
    return round(min_cavg, 4)

def compute_cprimary(trial_txt, score_txt, p_target_1=0.5, p_target_2=0.1):
    cprimary = compute_cavg(trial_txt, score_txt, p_target_1) + compute_cavg(trial_txt, score_txt, p_target_2)
    cprimary = cprimary/2
    return cprimary

if __name__ == "__main__":
    import subprocess
    eer_txt = '/home/hexin/Desktop/hexin/datasets/eer_3s.txt'
    score_txt = '/home/hexin/Desktop/hexin/datasets/score_3s.txt'
    trial_txt = '/home/hexin/Desktop/hexin/datasets/trial_3s.txt'
    subprocess.call(f"/home/hexin/Desktop/hexin/kaldi/egs/subtools/computeEER.sh "
                    f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
