FILESEXTRAPATHS:prepend := "${THISDIR}/${PN}:"

SRC_URI:append = " file://bsp.cfg"
KERNEL_FEATURES:append = " bsp.cfg"
SRC_URI += "file://user_2024-11-27-05-40-00.cfg \
            file://user_2024-11-29-03-33-00.cfg \
            "

