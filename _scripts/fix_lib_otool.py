import sys
import subprocess
import re


def find_libraries(filepath, filter_regex=False):
	out = subprocess.check_output(["otool", "-L", filepath])
	out = [line.strip() for line in out.split("\n")]

	if filter_regex:
		out = filter(lambda line:re.search(filter_regex, line), out)
		
	return out


def appendFilename(filepath, appendix):
	temp = filepath.split("/")
	temp[-1] = temp[-1].split(".")
	temp[-1][-2] = temp[-1][-2] + appendix
	temp[-1] = ".".join(temp[-1])
	return "/".join(temp)


if __name__ == "__main__":
	libPath = "../hstpl/extern_feat/libhesaff.dylib"
	localPath = "locallibcopy/"
	target = "opencv"

	libPathNew = appendFilename(libPath, "_NEW")
	lines = find_libraries(libPath, target)
	lines = [line.split(" ")[0] for line in lines]

	subprocess.call(["mkdir", localPath])
	subprocess.call(["cp", libPath, libPathNew])

	for line in lines:
		filename = line.split("/")[-1]
		subprocess.check_output(["cp", line, localPath + filename])
		subprocess.check_output(["install_name_tool", "-change", line, "@loader_path/" + localPath + filename, libPathNew])
	

