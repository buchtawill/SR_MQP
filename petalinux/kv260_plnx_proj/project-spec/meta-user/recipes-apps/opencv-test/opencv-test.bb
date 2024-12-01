#
# This file is the opencv-test recipe.
#

SUMMARY = "Simple opencv-test application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://opencv-test.cpp \
           file://Makefile \
		  "

DEPENDS += "opencv"

S = "${WORKDIR}"

do_compile() {
	     oe_runmake
}

do_install() {
	     install -d ${D}${bindir}
	     install -m 0755 opencv-test ${D}${bindir}
}
