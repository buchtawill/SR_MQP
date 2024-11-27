DESCRIPTION = "Include custom files for the root filesystem"
SECTION = "examples"
LICENSE = "CLOSED"
SRC_URI = "file://bitstream.bit"

do_install() {
    install -d ${D}/home/petalinux/
    install -m 0644 ${WORKDIR}/bitstream.bit ${D}/home/petalinux/
}
