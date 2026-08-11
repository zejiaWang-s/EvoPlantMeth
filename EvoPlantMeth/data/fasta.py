from __future__ import division, print_function
import os
from glob import glob
import gzip as gz
from ..utils import to_list

class FastaSeq(object):
    def __init__(self, head, seq):
        self.head = head
        self.seq = seq

def parse_lines(lines):
    seqs, start = [], None
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    for i, line in enumerate(lines):
        if line[0] == '>':
            if start is not None:
                seqs.append(FastaSeq(lines[start], ''.join(lines[start + 1: i])))
            start = i
    if start is not None:
        seqs.append(FastaSeq(lines[start], ''.join(lines[start + 1:])))
    return seqs

def read_file(filename, gzip=None):
    if gzip is None: gzip = filename.endswith('.gz')
    if gzip:
        lines = gz.open(filename, 'r').read().decode().splitlines()
    else:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
    return parse_lines(lines)

def select_file_by_chromo(filenames, chromo):
    filenames = to_list(filenames)
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        # Try multiple standard filename patterns
        patterns = [
            os.path.join(filenames[0], '*.chromosome.%s.fa*' % chromo),
            os.path.join(filenames[0], '*.chromosome.Chr%s.fa*' % chromo),
            os.path.join(filenames[0], '*.chromosome.chr%s.fa*' % chromo),
            os.path.join(filenames[0], '*.chr%s.fa*' % chromo),
            os.path.join(filenames[0], '%s.fa*' % chromo),
            os.path.join(filenames[0], 'chr%s.fa*' % chromo)
        ]
        for pattern in patterns:
            matched = glob(pattern)
            if matched: return matched[0]
        
        # Fallback to generic patterns if not found
        for filename in glob(os.path.join(filenames[0], '*.fa*')):
            base_name = os.path.basename(filename)
            if chromo in base_name or f"chr{chromo}" in base_name or f"Chr{chromo}" in base_name:
                return filename

    # Check filename directly if input is not a directory
    for filename in filenames:
        base_name = os.path.basename(filename)
        # Use flexible pattern matching for individual files
        if (f".chromosome.{chromo}." in base_name or
            f".chromosome.Chr{chromo}." in base_name or
            f".chromosome.chr{chromo}." in base_name or
            f".chr{chromo}." in base_name or
            f".{chromo}." in base_name or
            f"chr{chromo}.fa" in base_name or
            f"{chromo}.fa" in base_name):
            return filename
    return None

def read_chromo(filenames, chromo):
    filename = select_file_by_chromo(filenames, chromo)
    if not filename:
        # Try removing or adding 'chr' prefix as a final fallback
        alt_chromo = chromo[3:] if chromo.startswith('chr') else 'chr' + chromo
        filename = select_file_by_chromo(filenames, alt_chromo)
        if not filename:
            raise ValueError('DNA file for chromosome "%s" not found!' % chromo)

    fasta_seqs = read_file(filename)
    if len(fasta_seqs) != 1:
        raise ValueError('Single sequence expected in file "%s"!' % filename)
    return fasta_seqs[0].seq