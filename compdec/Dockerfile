FROM tensorflow/tensorflow:2.4.0

RUN addgroup --gid 40803 compdec && \
    adduser --uid 40803 --ingroup compdec --no-create-home --quiet --disabled-password --gecos "" compdec

WORKDIR /etc/compdec

USER compdec
COPY . .

ENTRYPOINT ["python3", "./compdec.py"]
