import struct
import array
import ctypes

def bytes2int8(bytes):
	''' Little endian parser which takes 1 unsigned integer and converts it to a single signed char. '''
	a = array.array('B', bytes)
	return struct.unpack('b', a)[0]

def bytes2uint8(bytes):
	''' Little endian parser which takes 1 unsigned integer and converts it to a single unsigned char. '''
	a = array.array('B', bytes)
	return struct.unpack('B', a)[0]

def bytes2int16(bytes):
	''' Little endian parser which takes a python list of 2 unsigned integers and converts them to a single signed short. '''
	a = array.array('B', bytes)
	return struct.unpack('h', a)[0]

def bytes2uint16(bytes):
	''' Little endian parser which takes a python list of 2 unsigned integers and converts them to a single unsigned short. '''
	a = array.array('B', bytes)
	return struct.unpack('H', a)[0]

def bytes2int32(bytes):
	''' Little endian parser which takes a python list of 4 unsigned integers and converts them to a single signed int. '''
	a = array.array('B', bytes)
	return struct.unpack('i', a)[0]

def bytes2uint32(bytes):
	''' Little endian parser which takes a python list of 4 unsigned integers and converts them to a single unsigned int. '''
	a = array.array('B', bytes)
	return struct.unpack('I', a)[0]

def bytes2int64(bytes):
	''' Little endian parser which takes a python list of 8 unsigned integers and converts them to a single long. '''
	a = array.array('B', bytes)
	return struct.unpack('l', a)[0]

def bytes2uint64(bytes):
	''' Little endian parser which takes a python list of 8 unsigned integers and converts them to a single unsigned long. '''
	a = array.array('B', bytes)
	return struct.unpack('L', a)[0]   #pandas: .astype(ctypes.c_ulong)

def bytes2float(bytes):
	''' Little endian parser which takes a python list of 4 unsigned integers and converts them to a single float. '''
	a = array.array('B', bytes)
	return struct.unpack('f', a)[0]

def bytes2double(bytes):
	''' Little endian parser which takes a python list of 8 unsigned integers and converts them to a single double. '''
	a = array.array('B', bytes)
	return struct.unpack('d', a)[0]

def bytes2bool(bytes):
	''' Little endian parser which takes a python list of 1 unsigned integers and converts them to a single boolean. '''
	a = array.array('B', bytes)
	return struct.unpack('?', a)[0]

def bytes2hexstring(bytes):
	"Return series of bytes as a hex string"
	return ' '.join(map(lambda x: hex(x), bytes))

def bytes2type(type, bytes=None):
	''' Takes a type string (see code below for options) and bytes (optional) and returns the bytes 
	converted to the given type. If bytes is not provided, the conversion function itself is returned
	instead.''' 

	typeConversionFunctions = \
	{ \
		'UINT8': bytes2uint8, \
		'INT8': bytes2int8, \
		'UINT16': bytes2uint16, \
		'INT16': bytes2int16, \
		'UINT32': bytes2uint32, \
		'INT32': bytes2int32, \
		'UINT64': bytes2uint64, \
		'INT64': bytes2int64, \
		'FLOAT': bytes2float, \
		'DOUBLE': bytes2double, \
		'BOOL': bytes2bool, \
		'HEXSTRING': bytes2hexstring \
	}

	type2Compare = type.strip().upper()

	try:
		func = typeConversionFunctions[type2Compare]
	except:
		print "No such type defined for: '" + str(type2Compare) + "'"
		raise

	if bytes:
		return func(bytes)
	else:
		return func