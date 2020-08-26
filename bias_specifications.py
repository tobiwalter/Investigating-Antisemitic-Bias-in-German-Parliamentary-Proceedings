## Target sets 

JEWISH_RT = ["jude", "juedisch", "judentum", "orthodox", "israel", "mosaisch","israelitisch","israelit", "rothschild", "talmud", "synagoge", "abraham", "rabbiner", "zionistisch"]

JEWISH_BRD = 'judentum, jude, juedisch, israel, israelisch, synagoge, koscher, orthodox, rabbiner, zentralrat'.split(', ')

CHRISTIAN_RT = ["christ", "christlich", "christentum", "katholizismus", "katholisch", "evangelisch", "evangelium", "auferstehung", "kirche" , "jesus", "taufe", "pfarrer", "bibel", "ostern"]

CHRISTIAN_BRD = 'christ, christlich, christentum, evangelisch, evangelium, jesus, katholisch, kirche, pfarrer, taufe, abendland'.split(', ')

PROTESTANT_BRD = "protestant, protestantisch, evangelisch, evangelium, landeskirche, kirchentag, ekd, landesbischof, lutherisch, diakonie".split(', ')

PROTESTANT_RT = ["protestant", "protestantisch", "protestantismus", "evangelisch", "evangelium", "landeskirche", "lutherisch", "evangelisch-lutherisch", "oberkirchenrat", "reformiert"]

CATHOLIC_BRD = "katholisch, katholik, papst, roemisch-katholisch, enzyklika, paepstlich, bischofskonferenz, dioezese, franziskus, kurie".split(', ')

CATHOLIC_RT = ["katholizismus", "katholisch", "katholik", "papst", "jesuiten", "ultramontanismus", "ultramontanen", "jesuitenorden", "roemisch-katholisch", "zentrumspartei"]

## Attribute sets defining different attribute dimensions

def antisemitic_streams(domain: str, kind: str):
	"""
	Create bipolar antisemitic stream of specified domain
	
	Parameters
	----------

	domain: Semantic domain to be created
  protocol_type: whether to process a collection of Reichstag or BRD protocols
	"""

	if domain == 'sentiment':
		pro = 'streicheln, Freiheit, Gesundheit, Liebe, Frieden, Freude, Freund, Himmel, loyal, Vergnuegen, Diamant, sanft, ehrlich, gluecklich, Regenbogen, Diplom, Geschenk, Ehre, Wunder, Sonnenaufgang, Familie, Lachen, Paradies, Ferien'.lower().split(', ') 

		con = 'Missbrauch, Absturz, Schmutz, Mord, Krankheit, Tod, Trauer, vergiften, stinken, Angriff, Katastrophe, Hass, verschmutzen, Tragoedie, Scheidung, Gefaengnis, Armut, haesslich, Krebs, toeten, faul, erbrechen, Qual'.lower().split(', ') 

	# nationalism/patriotism
	if domain == 'patriotism':
		if kind == 'RT':
			pro = 'patriotisch, germanisch, vaterlaendisch, deutschnational, reichstreu, vaterlandsliebe, nationalgesinnt, nationalstolz, koenigstreu, volksgeist, nationalbewusstsein, volksbewusstsein, staatstreu, nationalgefuehl'.split(', ')
			con = 'unpatriotisch, undeutsch, vaterlandslos, antideutsch, dissident, landesverraeter, reichsfeindlich, reichsfeind, deutschfeindlich, fremd, fremdlaendisch, nichtdeutsch, staatsfeindlich, heimatlos'.split(', ')

		elif kind == 'BRD':
			pro = 'patriotisch, vaterlandsliebe, germanisch, nationalbewusstsein, vaterlaendisch, nationalgefuehl, volkstum, patriotismus, patriot, staatstreu'.split(', ')

			con = 'nichtdeutsch, vaterlandslos, landesverraeter, antideutsch, heimatlos, separatistisch, staatsfeindlich, fremd, staatenlos, dissident'.split(', ')   

	# economic
	elif domain == 'economic':
		pro = 'geben, großzuegigkeit, großzuegig, selbstlos, genuegsam, großmut, uneigennuetzig, sparsam, proletariat, armut, industriearbeiter'.split(', ')

		con = 'nehmen, gier, gierig, egoistisch, habgierig, habsucht, eigennuetzig, verschwenderisch, bourgeoisie, wohlstand, bankier, wucher'.split(', ')

	# conspiracy
	elif domain == 'conspiratorial':
		pro = 'loyal, kamerad, ehrlichkeit, ersichtlich, aufrichtig, vertrauenswuerdig, wahr, ehrlich, unschuldig, freundschaftlich, hell, zugaenglich, machtlos, ohnmacht, untertan'.split(', ')

		con = 'illoyal, spitzel, verrat, geheim, betruegerisch, hinterlistig, unwahr, zweifelhaft, verbrecher, bedrohlich, dunkel, geheimnis, einflussreich, weltmacht, herrschaft, verschwoerung'.split(', ')

	# ethics
	elif domain == 'ethic':
		pro = 'bescheiden, sittlich, anstaendig, tugendhaft, charakterfest, wuerdig, treu, moralisch, ehrlich, gesittet, gewissenhaft, vorbildlich'.split(', ')
		con = 'unbescheiden, unsittlich, unanstaendig, luestern, korrupt, unwuerdig, untreu, unmoralisch, unehrlich, verdorben, gewissenlos, barbarisch'.split(', ')

	# religion
	elif domain == 'religious':
		pro = 'glaeubige, geistlich, engel, heilig, fromm, geheiligt, goettlich, ehrwuerdig, treu, glaeubig, religioes'.split(', ')

		con = 'atheist, weltlich, teufel, irdisch, atheistisch, heidnisch, gottlos, verflucht, untreu, unglaeubig, irreligioes, gotteslaesterung'.split(', ')

	# racism
	elif domain == 'racist':
		pro = 'normal, ueberlegenheit, gleichheit, angenehm, freundlich, ehrenwert, sympathie, akzeptiert, besser, national, rein, ueberlegen, sauber, ehrenhaft'.split(', ')
		con = 'seltsam, unterlegenheit, ungleichheit, unangenehm, boshaft, schaendlich, hass, abgelehnt, schlechter, fremdlaendisch, unrein, unterlegen, schmutzig, verseucht, schaedlich, niedertraechtig'.split(', ')

	return pro,con

