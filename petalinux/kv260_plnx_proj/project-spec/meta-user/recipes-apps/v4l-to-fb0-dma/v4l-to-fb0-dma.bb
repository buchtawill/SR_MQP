#
# This file is the v4l-to-fb0-dma recipe.
#

SUMMARY = "Simple v4l-to-fb0-dma application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://v4l-to-fb0-dma.cpp \
			file://axi-dma.cpp \
			file://axi-dma.h \
           file://Makefile \
		  "

S = "${WORKDIR}"

do_compile() {
	     oe_runmake
}

do_install() {
	     install -d ${D}${bindir}
	     install -m 0755 v4l-to-fb0-dma ${D}${bindir}
}
