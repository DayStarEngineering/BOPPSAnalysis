#!/usr/bin/python

# Imports:
import sys
import argparse
import operator
import packetdefinition
import pandas as pd
import datatypes

def readByteFromFile(fname):
	'''read byte from binary file

    Args:
        fname:  filename of the binary file
    
    Returns:
        byte: a single byte read from the file

    Examples:
        for byte in readByteFromFile("/path/to/file.bin"):
        	# do things

    Description:
    This generator opens up a binary file for reading and 
	yields one byte at a time until the entire file has been 
	read'''

	with open(fname, "rb") as f:
	    byte = f.read(1)
	    while byte:
	    	yield ord(byte)
	        byte = f.read(1)

def readPacketFromFile(fname):
	'''read sbpp packet from binary file

    Args:
        fname:  filename of the binary file
    
    Returns:
        packet: a single packet read from the file

    Examples:
        for packet in readPacketFromFile("/path/to/file.bin"):
        	# do things

    Description:
    This generator opens and reads a binary file, parsing and
	yielding one sbpp packet from the file at a time. If an invalid
	packet is found, it is ignored, and not returned'''

	def calculateChecksum(byteList):
		'''Calculate the sbpp packet checksum, an xor of the entire packet'''
		return reduce(operator.xor, byteList)

	byteGenerator = readByteFromFile(fname)
	for byte in byteGenerator:
		#print chr(byte)
		# Sense the start of a packet
		if byte == ord('B'):
			byte = byteGenerator.next()
			if byte == ord('R'):
				byte = byteGenerator.next()
				if byte == ord('R'):
					byte = byteGenerator.next()
					if byte == ord('I'):
						byte = byteGenerator.next()
						if byte == ord('S'):
							byte = byteGenerator.next()
							if byte == ord('O'):
								byte = byteGenerator.next()
								if byte == ord('N'):
									byte = byteGenerator.next()
									if byte == ord(':'):

										# Get length:
										data = []
										for i in xrange(4):
											byte = byteGenerator.next()
											data.append(byte)
										length = datatypes.bytes2uint32(data)

										# Get data:
										data = []
										for i in xrange(length):
											byte = byteGenerator.next()
											data.append(byte)

										yield data

# Main:
def parse(binaryFile, packetDefinitionFile):
	''' Takes a BOPPS binary file name and a packet definition filename and returns 
	a pandas dataframe containing all the parsed binary data for which a valid specification
	exists in the packet definition file. '''

	# Read and parse the packet definition file to obtain a dictionary of packet parsing functions:
	packetParser = packetdefinition.readPacketDefinitionFromFile(packetDefinitionFile)
	if not packetParser:
		print "reading the packet definition file '" + str(packetDefinitionFile) + "' failed!"
		raise

	# Read packets from file and parse each one in turn:
	parsedpackets = []
	for packet in readPacketFromFile(binaryFile):
		# Form a list of dictionaries for each packet:
		parsedPacket = packetParser(packet)
		if parsedPacket:
			parsedpackets.append(parsedPacket)

	# Combine parsed packets together into single pandas dataframe:
	df = pd.DataFrame(parsedpackets)

	return df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This utility opens and parses a binary controllaw file into a pandas data frame.')
	parser.add_argument('config', metavar='configfile', type=str, help='a SBPP packet definition file to ingest')
	parser.add_argument('fname', metavar='binaryfile', type=str, help='a SBPP binary file to ingest')
	args = parser.parse_args()
	df = parse(binaryFile = args.fname, packetDefinitionFile = args.config)
	print df