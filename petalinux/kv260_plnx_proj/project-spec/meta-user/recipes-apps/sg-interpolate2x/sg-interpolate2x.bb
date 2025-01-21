#
# This file is the sg-interpolate2x recipe.
#

SUMMARY = "Simple sg-interpolate2x application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI =  "file://sg-interpolate2x.cpp \
			file://sg-interpolate2x.h \
            file://Makefile \
		    file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/axi-dma.cpp \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/axi-dma.h \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/phys-mman.h \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/phys-mman.cpp \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/PhysMem.h \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/argparse.hpp \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/dma-sg-bd.cpp \
			file://${petalinux_root}/kv260_plnx_proj/project-spec/meta-user/recipes-apps/common-src/dma-sg-bd.h \
		  "

S = "${WORKDIR}"

do_compile() {
	     oe_runmake
}

do_install() {
	     install -d ${D}${bindir}
	     install -m 0755 sg-interpolate2x ${D}${bindir}
}
