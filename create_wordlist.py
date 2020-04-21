# Create a word list from the Hunspell dictionary 

with open('dictionaries\\ger-wordlist', 'w') as fout:	
	with open('dictionaries\\hunspell_de.dic', 'r') as fin:
		for line in fin:
			word = line.split('/')[0]
			fout.write(word)
			fout.write('\n')

