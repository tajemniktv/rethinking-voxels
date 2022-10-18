fshchecks = ""
with open("shadowchecks_fsh.glsl") as checksfile:
	fshchecks = checksfile.read()
colnames = {}
j = 0
k = 0
i = 0
i0 = 0
hc = "HARDCODED_"
brightness = "BRIGHTNESS_"
channels = ["_R", "_G", "_B"]
currentname = []
brightnessvals = {}
currentlyInName = False
while i < len(fshchecks):
	if currentlyInName:
		if fshchecks[i] in " \n;":
			currentlyInName = False
			currentname = "".join(currentname)
			try:
				colnames[currentname].append(i0)
			except:
				colnames[currentname] = [i0]
		else:
			currentname.append(fshchecks[i])
	else:
		j = j + 1 if hc[j] == fshchecks[i] else 0
		k = k + 1 if brightness[k] == fshchecks[i] else 0
		if k == len(brightness):
			prefix = []
			n = 11
			while i >= n and not fshchecks[i - n] in " \n(":
				prefix.append(fshchecks[i - n])
				n += 1
			prefix = "".join(prefix[::-1])
			suffix = []
			n = 1
			while not fshchecks[i + n] in "; \n":
				suffix.append(fshchecks[i + n])
				n += 1
			suffix = "".join(suffix)
			try:
				if not prefix in brightnessvals[suffix + "_COL"]:
					brightnessvals[suffix + "_COL"].append(prefix)
			except:
				brightnessvals[suffix + "_COL"] = [prefix]
			k = 0
		if j == len(hc):
			currentlyInName = True
			j = 0
			i0 = i
			currentname = []
	i += 1
colOccurrences = {}
for n in colnames.keys():
	for i in colnames[n]:
		prefix = []
		k = 10
		while i >= k and not fshchecks[i - k] in " \n(":
			prefix.append(fshchecks[i - k])
			k += 1
		prefix = "".join(prefix[::-1])
		try:
			if not prefix in colOccurrences[n]:
				colOccurrences[n].append(prefix)
		except:
			colOccurrences[n] = [prefix]
definefile = []
shadersprop = ["screen.LIGHTCOLS="]
for n in colnames.keys():
	shadersprop[0] += "[" + n + "] "
	thisscreenentriesl = []
	thisscreenentriesr = []
	for o in colOccurrences[n]:
		thisscreenentriesl.append( o + hc + n)
		definefile.append("#define " + o + hc + n)
	for c in channels:
		thisscreenentriesl.append(n + c)
		definefile.append("#define " + n + c + " 1.0 //[0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0]")
	try:
		for v in brightnessvals[n]:
			thisscreenentriesr.append(v + brightness + n[:-4])
			definefile.append("#define " + v + brightness + n[:-4] + " 13 //[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]")
	except:
		print("No brightness values for " + n[:-4] + "-related colours!")
		True
	definefile.append("")
	thisscreenentries = []
	for k in range(max(len(thisscreenentriesl), len(thisscreenentriesr))):
		try:
			thisscreenentries.append(thisscreenentriesl[k])
		except:
			thisscreenentries.append("<empty>")
		try:
			thisscreenentries.append(thisscreenentriesr[k])
		except:
			thisscreenentries.append("<empty>")
	shadersprop.append("screen." + n + "=" + " ".join(thisscreenentries))
shadersprop[0] += "[OTHER_COL]"
thisscreenentries = []
brightnessvalsliders = []
for n in brightnessvals.keys():
	for v in brightnessvals[n]:
		brightnessvalsliders.append(v + brightness + n[:-4])
		if not n in colnames.keys():
			thisscreenentries.append(v + brightness + n[:-4])
			definefile.append("#define " + v + brightness + n[:-4] + " 13 //[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]")
shadersprop.append("screen.OTHER_COL=" + " ".join(thisscreenentries))
definefile = "\n".join(definefile)
with open("definefile.tmp", "w") as df:
	df.write(definefile)
print("additions to shaders.properties:\n" + "\n".join(shadersprop))
sliders = [n + c for n in colnames.keys() for c in channels] + brightnessvalsliders
print("New sliders:\n" + " ".join(sliders))
