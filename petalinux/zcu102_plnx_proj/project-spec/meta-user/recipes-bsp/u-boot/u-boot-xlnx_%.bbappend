FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI:append = " file://platform-top.h file://bsp.cfg"
SRC_URI += "file://user_2025-02-19-20-35-00.cfg \
            file://user_2025-02-19-21-40-00.cfg \
            "

