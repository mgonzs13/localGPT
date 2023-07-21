FROM ubuntu:22.04

USER root
ARG DEBIAN_FRONTEND=noninteractive

COPY ./src/install/utils.sh $INST_SCRIPTS/
RUN chmod +x $INST_SCRIPTS/utils.sh && $INST_SCRIPTS/utils.sh

COPY ./src/install/localGPT.sh $INST_SCRIPTS/
RUN chmod +x $INST_SCRIPTS/localGPT.sh && $INST_SCRIPTS/localGPT.sh

COPY ./src/entrypoint/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root

ENTRYPOINT ["/root/entrypoint.sh"]