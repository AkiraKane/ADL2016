import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--treefile", help="echo the string you use here")
args = parser.parse_args()

# Produce pos_tag list (these are used to remove the pos_tag in the tree files)
def produce_postag(filename):
    posfile = open(filename,"r").readlines()

    line_list = []
    temp_str = ""
    for line in posfile:
        if ", ," in line:
            line = line.replace(", ,",",")
        if ". ." in line:
            end = line.split(".")[0] + "." + line.split(".")[2].strip()
            temp_str += end
            line_list.append(temp_str)
            temp_str = ""
            continue
        temp_str += line.strip()

    pos_tag_list = [] 
    for line in line_list:
        temp_tag_list = []
        line_split = line.split("(")
        for split_sub in line_split:
            if split_sub is not "":
                tag = split_sub.split(" ")[0]
                temp_tag_list.append(tag)
        pos_tag_list.extend(temp_tag_list)

    pos_set = set(pos_tag_list)
    remove_list = [ '$',':', ',)))))))', '``', "''",'.)','.)))))','#', ',)))', ',)', '.)))', '.))))))',',))))', ":", ',))', '.))))', ',)))))))))','.))',","]
    pos_list = [ i for i in pos_set if i not in remove_list ]
    postag_list = sorted(pos_list, key=len, reverse=True)
    return postag_list


# tree file preprocess
def preprocess_treefile(filename, postag_list):
    posfile = open(filename,"r").readlines()
    outputfilename = "revised_tree_file.txt"
    outputfile = open(outputfilename,"w")

    line_list = []
    temp_str = ""
    for line in posfile:
        if ", ," in line:
            line = line.replace(", ,",",")
        if ". ." in line:
            end = line.split(".")[0] + "." + line.split(".")[2].strip()
            temp_str += end
            line_list.append(temp_str)
            temp_str = ""
            continue
        temp_str += line.strip()

    for line in line_list:
        for tag in postag_list:  
            line = line.replace(tag, "") 
        new_line = list(line.strip())
        new_line = [i for i in new_line if i is not " "]
        new_text = " ".join(new_line)
        outputfile.write(new_text)
        outputfile.write("\n")

    outputfile.close()

if __name__=="__main__":
    postag_list = ['WHADVP', 'WHADJP', '-LRB-', '-RRB-', 'SBARQ', 'CONJP', 'PRP$', 'SBAR', 'SINV', 'ADJP', 'WHPP', 'FRAG', 'INTJ', 'ADVP', 'NNPS', 'WHNP', 'VBG', 'VBN', 'JJR', 'VBP', 'WDT', 'SYM', 'VBZ', 'PRN', 'VBD', 'RRC', 'POS', 'PRT', 'LST', 'PRP', 'NNS', 'NNP', 'WRB', 'PDT', 'RBS', 'RBR', 'WP$', 'UCP', 'JJS', 'FW', 'JJ', 'WP', 'DT', 'PP', 'RP', 'NN', 'TO', 'RB', 'NP', 'VB', 'QP', 'CC', 'LS', 'CD', 'VP', 'EX', 'IN', 'MD', 'SQ', 'UH', 'S', 'X']
    
    test_filename = args.treefile
    preprocess_treefile(test_filename, postag_list)
