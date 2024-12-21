#
# This file is the v4l-to-fb0-aligned recipe.
#

SUMMARY = "Simple v4l-to-fb0-aligned application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://v4l-to-fb0-aligned.cpp \
           file://Makefile \
		  "

S = "${WORKDIR}"

do_compile() {
	     oe_runmake
}

do_install() {
	     install -d ${D}${bindir}
	     install -m 0755 v4l-to-fb0-aligned ${D}${bindir}
}
