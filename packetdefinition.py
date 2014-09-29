#!/usr/bin/python
import sys
import datatypes
import argparse

def parsePacket(packetDefinition):
	''' dynamic parsing function generator

    Args:
        packetDefinition:  a list of dictionaries of the form:
			[{'name': valueName, 'typeFunc': funcWhichConvertsType, 'modifyFunc': funcWhichModifiesVal, \
			 'startByte': indexWhereValueStarts, 'stopByte': indexWhereValueEnds}, ...nextValueDictionary...]
    
    Returns:
        parse:  A dynamically generated parsing function which takes packet the data field (python list of bytes) as input

    Examples:
        parseFunc = parsePacket([{'name': 'Position X', 'typeFunc': myTypeFunction1, 'modifyFunc': myLambdaFunction1, 'startByte': 0, 'stopByte': 4}, \
        	                     {'name': 'Position Y', 'typeFunc': myTypeFunction2, 'modifyFunc': myLambdaFunction2, 'startByte': 4, 'stopByte': 8}])

    Description:
		This function is a closure which takes a packet definition and returns a parsing function which takes
		packet data and returns a parsed packet dictionary. The packet definition is a list of dictionaries of the form:

		[{'name': valueName, 'typeFunc': funcWhichConvertsType, 'modifyFunc': funcWhichModifiesVal, \
		 'startByte': indexWhereValueStarts, 'stopByte': indexWhereValueEnds}, ...nextValueDictionary...]

		The packet data input to the returned parse function is a list of single byte integer data with the header, sync, 
		footer, id, and length removed.

		The output of the parse function is a parsed packet dictionary in the form:
		{'DataValueName1': dataValue1, 'DataValueName2': dataValue2, ...etc...}

		This parsing function returned by parsePacket is dynamic based on the packetDefinition input. A static
		version of a returned "parse" function for packet 0x10 is shown below as an example of how the dynamic function 
		behaves:

		def parse10(data):
			return {'Position X': bytes2float(data[0:4]), 'Position Y': bytes2float(data[4:8]), 'Position Z': bytes2float(data[8:12]),  \
				    'Velocity X': bytes2float(data[12:16]), 'Velocity Y': bytes2float(data[16:20]), 'Velocity Z': bytes2float(data[20:24]), \
				    'GPS Week': bytes2uint16(data[24:26]), 'GPS Week': bytes2double(data[26:34]), 'Clock Bias': bytes2float(data[34:38]), \
				    'Clock Bias Rate': bytes2float(data[38:42]), 'Num Sats Used in Fix': bytes2uint8(data[42]), 'GDOP': bytes2uint8(data[43]), \
				    'Position Fix Validity': bytes2uint8(data[44])}
	'''
	def parse(data):
		parsedPacket = {}
		for definition in packetDefinition:
				parsedPacket[definition['name']] = definition['modifyFunc']( \
												        definition['typeFunc']( \
												        	data[definition['startByte']:definition['endByte']] \
												        	) \
												        )
		return parsedPacket
	return parse

def readLineFromFile(fname):
	'''read line from text file

    Args:
        fname:  filename to open and begin reading lines from
    
    Returns:
        line: a string representing the current line read

    Examples:
        for line in readLineFromFile("/path/to/file.txt"):
        	# do things

    Description:
    This generator opens up a file fname and returns the text inside line by line'''
	with open(fname, "r") as f:
	    for line in f:
	    	yield line

def readPacketDefinitionFromFile(fname):
	''' read and parse packet definition file

    Args:
        fname:  filename of packet definition file
    
    Returns:
        packetParsers: a packet definition dictionary of the form {packetId1: <parserFuncForId1>, packetId4: <parserFuncForId4>, ...etc...}

    Examples:
        parserFunctions = readPacketDefinitionFromFile("/path/to/packetdefinition.txt")

    Description:
    This function takes a packet definition filename, opens it and parses the input. If the
	packet definition file is malformed, then the line with the error on it is printed and the
	program is terminated. If the file is well formed, a packetDefinition dictionary is returned
	in the form:

	{packetId1: <parserFuncForId1>, packetId4: <parserFuncForId4>, ...etc...}

	where packetIdN is an integer representing the packet id for which the python function parserFuncForIdN
	has been generated to parse packets of that type, according to the specification outlined in the 
	packet definition file '''

	def line2ValueDefinition(splitline):
		'''Takes a split line from the packet definition file and formulates a valueDefinition
		dictionary from that line in the form:

		{'startByte': integerStartByte, 'endByte': integerEndByte, 'name': valueName, \
		'typeFunc': typeConversionFunctionHandle, 'modifyFunc': modifyValueFunctionHandle}
		'''
		valueDefinition = {}
		startAndEndBytes = splitline[0].split('-')

		try:
			startByte = int(startAndEndBytes[0].strip())
		except:
			startByte = None

		# Handle end byte being none, or not being present:
		if len(startAndEndBytes) < 2:
			endByte = startByte + 1
		else:
			try:
				endByte = int(startAndEndBytes[1].strip()) + 1
			except:
				endByte = None

		valueDefinition['startByte'] = startByte
		valueDefinition['endByte'] = endByte
		valueDefinition['name'] = splitline[1]
		valueDefinition['typeFunc'] = datatypes.bytes2type(splitline[2])

		if len(splitline) >= 4 and splitline[3]:
			valueDefinition['modifyFunc'] = eval("lambda x: " + str(splitline[3]))
		else:
			valueDefinition['modifyFunc'] = lambda x: x

		return valueDefinition

	packetDefinitions = []
	linenum = 0
	for line in readLineFromFile(fname):
		linenum += 1
		line = line.strip()
		if line:
			try:
				# Handle comments:
				if line[0] == '#':
					continue

				# Split the line and remove white space:
				splitline = line.split(",")
				splitline = map(lambda x: x.strip(), splitline)

				# Handle normal line:
				if len(splitline) >= 3:
					valueDefinition = line2ValueDefinition(splitline)
					if valueDefinition:
						packetDefinitions.append(valueDefinition)
				else:
					print "Invalid configuration entry:", line
			except:
				print "Error in configuration file on line:", linenum, "('" + line + "')"
				raise

	# Create packet parser function:
	func = parsePacket(packetDefinitions)
	return func

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This utility opens and parses a packet configuration file.')
	parser.add_argument('config', metavar='configfile', type=str, help='a SBPP packet definition file to ingest')
	args = parser.parse_args()
	sys.exit(readPacketDefinitionFromFile(fname = args.config))