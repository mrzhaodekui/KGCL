import numpy as np
import pandas as pd
import os
import argparse
import joblib
from tqdm import tqdm
from collections import Counter
import torch
from rdkit import Chem, RDLogger

from models import KGCL, BeamSearch

import sys

lg = RDLogger.logger()
lg.setLevel(4)

ROOT_DIR = './'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [
            y.ClearProp('molAtomMapNumber') for y in mol.GetAtoms()
            if y.HasProp('molAtomMapNumber')
        ]
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def canonicalize_smiles_clear_map(smiles, return_max_frag=True):

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        [
            atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()
            if atom.HasProp('molAtomMapNumber')
        ]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '', ''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [
                Chem.MolFromSmiles(smiles)
                for smiles in sub_smi
            ]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms()))
                            for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(
                    sorted(sub_mol_size, key=lambda x: x[1],
                           reverse=True)[0][0],
                    return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '', ''
        else:
            return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or USPTO_full or uspto_mit')
    parser.add_argument("--use_rxn_class", default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--experiments', type=str, default='BEST',
                        help='Name of edits prediction experiment')
    parser.add_argument('--beam_size', type=int,
                        default=50, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_file = os.path.join(data_dir, 'test.file.kekulized')
    test_data = joblib.load(test_file)
    if args.use_rxn_class:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')
        checkpoint = 'epoch_128.pt'
    else:
        exp_dir = os.path.join(
            ROOT_DIR, 'experiments', f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')
        checkpoint = 'epoch_132.pt'

    checkpoint = torch.load(os.path.join(exp_dir, checkpoint))
    config = checkpoint['saveables']

    model = KGCL(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    MaxFrag_top_k = np.zeros(args.beam_size)
    edit_steps_cor = []
    counter = []
    stereo_rxn = []
    stereo_rxn_cor = []
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)
    p_bar = tqdm(list(range(len(test_data))))
    pred_file = os.path.join(exp_dir, 'pred_results.txt')


    with open(pred_file, 'a') as fp:
        for idx in p_bar:
            rxn_data = test_data[idx]
            rxn_smi = rxn_data.rxn_smi
            rxn_class = rxn_data.rxn_class
            edit_steps = len(rxn_data.edits)
            counter.append(edit_steps)

            r, p = rxn_smi.split('>>')
            r_mol = Chem.MolFromSmiles(r)
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]
            r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
            r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)
            r_set = set(r_smi.split('.'))

            r_c, r_maxfrag = canonicalize_smiles_clear_map(r)

            with torch.no_grad():
                top_k_results= beam_model.run_search(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)

            fp.write(f'({idx}) {rxn_smi}\n')

            beam_matched = False
            MaxFrag_beam_matched = False
            for beam_idx, path in enumerate(top_k_results):
                pred_smi = path['final_smi']
                prob = path['prob']
                pred_set = set(pred_smi.split('.'))
                correct = pred_set == r_set
                pred_c, pred_MaxFrag = canonicalize_smiles_clear_map(pred_smi)
                MaxFrag_correct = pred_MaxFrag == r_maxfrag
                str_edits = '|'.join(f'({str(edit)};{p})'for edit, p in zip(
                    path['rxn_actions'], path['edits_prob']))
                fp.write(
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')

                if correct and not beam_matched:
                    top_k[beam_idx] += 1
                    beam_matched = True
                if MaxFrag_correct and not MaxFrag_beam_matched:
                    MaxFrag_top_k[beam_idx] +=1
                    MaxFrag_beam_matched = True

            fp.write('\n')
            if beam_matched:
                edit_steps_cor.append(edit_steps)

            for edit in rxn_data.edits:
                if edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (2, 2) or edit[1] == (2, 3):
                    stereo_rxn.append(idx)
                    if beam_matched:
                        stereo_rxn_cor.append(idx)

            msg = 'average score'
            for beam_idx in [1, 3, 5, 10, 50]:
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)
                MaxFrag_match_acc = np.sum(MaxFrag_top_k[:beam_idx]) / (idx + 1)
                msg += ', t%d: %.3f' % (beam_idx, match_acc)
                msg += ', MaxFrag:' + 't%d: %.3f' % (beam_idx, MaxFrag_match_acc)
            p_bar.set_description(msg)

        edit_steps = Counter(counter)
        edit_steps_correct = Counter(edit_steps_cor)
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')
        fp.write(
            f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))


if __name__ == '__main__':
    main()