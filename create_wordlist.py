# Create a word list from the Hunspell dictionary 

with open('dictionaries\\ger-wordlist.txt', 'w') as fout:	
	with open('dictionaries\\de_DE\\German_de_DE.dic', 'r') as fin:
		for line in fin:
			if '/' in line:
				word = line.split('/')[0]
				fout.write(word)
				fout.write('\n')
			else:
				fout.write(line)
	
