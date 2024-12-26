#
# This file is the v4l-to-fb0-dma recipe.
#

SUMMARY = "Simple v4l-to-fb0-dma application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

# WORKDIR is SR_MQP/petalinux/kv260_plnx_proj/build/tmp/work/cortexa72-cortexa53-xilinx-linux/v4l-to-fb0-dma/1.0-r0

SRC_URI = "file://v4l-to-fb0-dma.cpp \
			file://v4l-to-fb0-dma.h \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/axi-dma.cpp \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/axi-dma.h \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/dma-sg-bd.h \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/dma-sg-bd.cpp \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/phys-mman.h \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/phys-mman.cpp \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/PhysMem.h \
			file://${WORKDIR}/../../../../../../project-spec/meta-user/recipes-apps/common-src/bits.h \
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
