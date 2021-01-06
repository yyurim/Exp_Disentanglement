#!/usr/bin/env python3
#coding=utf8
import os
import numpy as np
import kaldi_io

###########################################################################
def get_specifier(file_name, option=[]):
    ext = file_name.split(".")[-1]
    if ext not in ["scp", "ark"]:
        ext = "ark"
    specifier =  ext + ":" + file_name
    return specifier

def make_kaldi_cmd(kaldi_cmd, *args):
    result_cmd = kaldi_cmd +" " + " ".join(args) + " |"
    return result_cmd

def copy_feats(feat_path="-", out_path="-"):
    """
    copy-feats <ark,scp:feats.ark,scp> <ark:-> |
    """
    kaldi_cmd="copy-feats"
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    return result_cmd

def copy_vector(vec_path="-", out_path="-"):
    """
    copy-vector <ark,scp:feats.ark,scp> <ark:-> |
    """
    kaldi_cmd="copy-vector"
    r_specifier=get_specifier(vec_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    return result_cmd

def compute_cmvn_stats(feat_path="-", out_path="-", spk2utt_path="-"):
    """
    compute-cmvn-stats <ark:feats.ark> <ark:-> |
    """
    kaldi_cmd="compute-cmvn-stats"
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    if spk2utt_path is "-":
        result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    else:
        s2u_specifier=get_specifier(spk2utt_path)
        result_cmd=make_kaldi_cmd(kaldi_cmd, "--spk2utt="+s2u_specifier, r_specifier, o_specifier)
    return result_cmd

def apply_cmvn(cmvn_path="-", feat_path="-", out_path="-", utt2spk_path="-"):
    """
    apply-cmvn <ark:feats.scp> <ark:-> |
    """
    kaldi_cmd="apply-cmvn"
    r_specifier1=get_specifier(cmvn_path)
    r_specifier2=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    if utt2spk_path is "-":
        result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier1, r_specifier2, o_specifier)
    else:
        u2s_specifier=get_specifier(utt2spk_path)
        result_cmd=make_kaldi_cmd(kaldi_cmd, "--utt2spk="+u2s_specifier, r_specifier1, r_specifier2, o_specifier)
    return result_cmd
def add_deltas(feat_path="-", out_path="-"):
    """
    add-deltas <scp:feats.scp> <ark:-> |
    """
    kaldi_cmd="add-deltas"
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, r_specifier, o_specifier)
    return result_cmd
def gunzip(ali_path):
    """
    gunzip -c <ali.1.gz> |
    """
    kaldi_cmd = "gunzip"
    option="-c"
    result_cmd = make_kaldi_cmd(kaldi_cmd,option,ali_path)
    return result_cmd
def ali_to_pdf(mdl_path, ali_path="-", out_path="-"):
    """
    ali-to-pdf <final.mdl> <ark:ali.1.ark> <ark:-> |
    """
    kaldi_cmd="ali-to-pdf"
    r_specifier = get_specifier(ali_path)
    o_specifier = get_specifier(out_path)
    result_cmd=make_kaldi_cmd(kaldi_cmd, mdl_path, r_specifier, o_specifier)
    return result_cmd
def splice_feats(left, right, feat_path="-", out_path="-"):
    """
    splice-feats <--left-context=5> <--right-context=5> <ark:feats.ark> <ark:-> |
    """
    kaldi_cmd="splice-feats"
    option1="--left-context="+left
    option2="--right-context="+right
    r_specifier=get_specifier(feat_path)
    o_specifier=get_specifier(out_path)
    result_cmd = make_kaldi_cmd(kaldi_cmd, option1, option2, r_specifier, o_specifier)
    return result_cmd
###########################################################################

class KaldiReadManager:
    """
    Read kaldi data from HDD by using assemblized kaldi command
    Note that read method is generator
    """
    def __init__(self):
        self.cmd = ""
        self.cmd_book = dict()
        self.init_command()
        self.init_command_book()
    
    def run(self):
        self.cmd = self.cmd[:-1] # delete |
        os.system(self.cmd)

    def init_command_book(self):
        """ need to fix
        store for kaldi command
        """
        self.cmd_book["copy-feats"] = copy_feats
        self.cmd_book["copy-vector"] = copy_vector
        self.cmd_book["apply-cmvn"] = apply_cmvn
        self.cmd_book["compute-cmvn-stats"] = compute_cmvn_stats
        self.cmd_book["add-deltas"] = add_deltas
        self.cmd_book["splice-feats"] = splice_feats
        self.cmd_book["gunzip"] = gunzip
        self.cmd_book["ali-to-pdf"] = ali_to_pdf

    def init_command(self):
        self.cmd = ""

    def set_command(self, command, *args, **kwargs):
        assert command in self.cmd_book, "wrong kaldi command"
        cur_command = self.cmd_book[command](*args, **kwargs)
        self.cmd += cur_command

    def read_to_mat(self):
        print("run",self.cmd)
        generator = kaldi_io.read_mat_ark(self.cmd)
        result = {utt_id: np.array(frame_mat) for utt_id, frame_mat in generator}
        return result

    def read_to_vec(self, type='int'):
        print("run",self.cmd)
        if type=='int':
            generator = kaldi_io.read_vec_int_ark(self.cmd)
        if type=='float':
            generator = kaldi_io.read_vec_flt_ark(self.cmd)
        result = {utt_id: np.array(vec) for utt_id, vec in generator}
        return result

def read_feat(feat_path, spk_cmvn=False, utt_cmvn=False, delta=True, linear=False):
    km = KaldiReadManager()
    feat_dir = "/".join(feat_path.split("/")[:-1])
    if spk_cmvn:
        km.set_command("compute-cmvn-stats", feat_path=feat_path, out_path=feat_dir+"/cmvn.ark", spk2utt_path=feat_dir+"/spk2utt")
        km.run()
    if utt_cmvn:
        km.set_command("compute-cmvn-stats", feat_path=feat_path, out_path=feat_dir+"/cmvn.ark")
        km.run()
    
    km.init_command()
    km.set_command("copy-feats", feat_path)
    if spk_cmvn:
        km.set_command("apply-cmvn", cmvn_path=feat_dir+"/cmvn.ark", utt2spk_path=feat_dir+"/utt2spk")
    if utt_cmvn:
        os.system("compute-cmvn-stats ark:"+feat_dir+"/feats.ark ark:"+feat_dir+"/cmvn.ark" )
        km.set_command("apply-cmvn", cmvn_path=feat_dir+"/cmvn.ark")
    if delta:
        km.set_command("add-deltas")
    feat_dict = km.read_to_mat()
    if linear:
        for utt_id, feat_mat in feat_dict.items():
            feat_dict[utt_id]=np.exp(feat_mat)
    return feat_dict

def read_ali(ali_root):
    """
    "gunzip -c ali/train/ali.*.gz | ali-to-pdf ali/train/final.mdl ark:- ark:- |"
    """
    ali_path = ali_root+"/ali.*.gz"
    mdl_path = ali_root+"/final.mdl"
    km = KaldiReadManager()
    km.set_command("gunzip", ali_path)
    km.set_command("ali-to-pdf", mdl_path)
    ali_dict = km.read_to_vec(type='int')
    return ali_dict

def read_vec(vec_path):
    km = KaldiReadManager()
    km.set_command("copy-vector", vec_path)
    ali_dict = km.read_to_vec(type='float')
    return ali_dict

def get_gmm_info(ali_path):
    os.system("gmm-info "+ali_path+"/final.mdl > ali.tmp")
    gmm_info = dict()
    with open("ali.tmp", 'r') as f:
        for line in f:
            line_info = line[:-1].split(" ")
            if line_info[0] != "number":
                continue
            else:
                gmm_info[line_info[-2]] = int(line_info[-1])
    os.system("rm ali.tmp")        
    return gmm_info

def get_pdf_num(ali_path):
    gmm_info = get_gmm_info(ali_path)
    return gmm_info["pdfs"]

def write_dict(ark_path, data_dict):
    fd = kaldi_io.open_or_fd(ark_path, mode='wb')
    for utt_id, mat in data_dict.items():
        kaldi_io.write_mat(fd, np.array(mat), utt_id)
    fd.close()
