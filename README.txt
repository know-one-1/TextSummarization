Checked on 
python 3.7 
pytorch 1.4.0

Data PreProcess :

1 . python3 Data_preparation.py --path='path/to/csv/file'
	
	[It will create three tar folders in the directory of code and a vocab file extract the train test and val folder in a folder and put vocab file also in that folder the tree should look like]

	data/directory
			--train
			--test
			--val
			--vaocab (file)

#####################################################################################################
[From 3 - 13 to  get results on TL;DR dataset]

3. pyrouge installation :

code :
	1. git clone https://github.com/bheinzerling/pyrouge

	2. cd pyrouge

	3. pip install -e .

	4. git clone https://github.com/andersjo/pyrouge.git rouge

	5. pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/

	6. sudo apt-get install libxml-parser-perl

	7. cd rouge/tools/ROUGE-1.5.5/data

	8. rm WordNet-2.0.exc.db

	9. ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

	10. python3 -m pyrouge.test

	11. export ROUGE='path/to/rouge/directory] , ( '~/pyrouge/rouge/tools/ROUGE-1.5.5/''this path])

4. export DATA='path/to/data/directory'
	 
5. python3 train_word2vec.py --path='path/to/save/word2vec'

6. python3 make_extraction_labels.py

7. python3 train_abstractor.py --path='path/to/save/abstractor/model' --w2v='path/to/word2vec/word2vec.128d.*k.bin(file)'

8. python3 train_extractor_ml.py --path='path/to/save/extractor/model] --w2v='path/to/word2vec/word2vec.128d.*k.bin(file)'

9. python3 train_full_rl.py --path='path/to/save/model] --abs_dir='path/to/abstractor/model] --ext_dir='path/to/extractor/model'

11. python3 decode_full_model.py --path='path/to/save/decoded/files] --model_dir='path/to/pretrained] --beam='beam_size] '--test/--val'
	Note : use beam=5 and --test

12. python3 make_eval_references.py

13. python3 eval_full_model.py --rouge --decode_dir='path/to/save/decoded/files'

##############################################################################################################

14. python3 decode_for_discrimi.py --path='path/to/save/decoded/files' --model_dir='path/to/RL/model/directory'

15. python3 CopyDiscriminator.py/Discriminator.py --path='path/to/save/model'

16. python3 copyDiscri_evaluate.py/Discri_evaluate.py --path='path/to/saved/dscriminator/model' 

		[just to check how the 	discriminator is working]

17.  python3 extracted_labels_RLextractor --path='/path/to/RL/model' [to get labels of extraction from trained extractor to 
		train generator]

18 . python3 Generator.py --path='path/model/to/be/saved' --model_dir='path/to/full/RL/model' 
			--dis_dir='path/to/discriminator/model' 
