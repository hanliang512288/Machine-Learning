require 'csv'

fromHash = {}
toHash = {}

File.open("train.txt", "r") do |file|
    while line = file.gets
        lineArray = line.split(/[\t\r\n]/)
        key = lineArray.shift
        if lineArray.length != 0
            fromHash[key] = lineArray
        end
    end
end

puts "end of read train original"

File.open("trainreverse.txt", "r") do |file|
    while line = file.gets
        lineArray = line.split(/[\t\r\n]/)
        key = lineArray.shift
        toHash[key] = lineArray
    end
end

puts "end of read train reverse"

subsetHash = {}

File.open("subset.txt", "r") do |file|
    while line = file.gets
        lineArray = line.split(/[\t\r\n]/)
        key = lineArray.shift
        subsetHash[key] = lineArray
    end
end

puts "end of read subset"

subsettoNode = []

File.open("subsetreverse.txt", "r") do |file|
    while line = file.gets
        lineArray = line.split(/[\t\r\n]/)
        subsettoNode << lineArray.shift
    end
end

puts "end of read subsetreverse"

fromNode = fromHash.keys
toNode = toHash.keys

subsetfromNode = subsetHash.keys

i = 0
j = 0
id = 0

CSV.open("result.csv", "w+") do |w|
    w << ["Id","from","to","fromout","fromin","toout","toin","fromrate","torate","width","range","cofrom","coto","exist"]

    while i < 500000
        sampleFrom = subsetfromNode[rand(subsetfromNode.length)]
        toArray = subsetHash[sampleFrom]
        sampleTo = toArray[rand(toArray.length)]

        if toHash[sampleFrom] == nil
            toHash[sampleFrom] = []
        end
        if fromHash[sampleTo] == nil
            fromHash[sampleTo] = []
        end

        id += 1
        fromout = fromHash[sampleFrom].length
        fromin = toHash[sampleFrom].length
        toout = fromHash[sampleTo].length
        toin = toHash[sampleTo].length

        fromrate = fromin.to_f / fromout
        torate = toout.to_f / toin

        width = (fromHash[sampleFrom] & toHash[sampleTo]).length

        range = 0
        middle = fromHash[sampleFrom]
        middle.each do |m|
            if fromHash[m] != nil
                range += fromHash[m].length
            end
        end

        intersection = (fromHash[sampleFrom] & fromHash[sampleTo]).length
        union = (fromHash[sampleFrom] | fromHash[sampleTo]).length
        cofrom = intersection.to_f / union

        intersection = (toHash[sampleFrom] & toHash[sampleTo]).length
        union = (toHash[sampleFrom] | toHash[sampleTo]).length
        coto = intersection.to_f / union

        exist = 1

        w << [id,sampleFrom,sampleTo,fromout,fromin,toout,toin,fromrate,torate,width,range,cofrom,coto,exist]

        i += 1
    end

    while j < 500000
        sampleFrom = subsetfromNode[rand(subsetfromNode.length)]
        sampleTo = subsettoNode[rand(subsettoNode.length)]

        if sampleFrom != sampleTo
            if !fromHash[sampleFrom].include?sampleTo
                if toHash[sampleFrom] == nil
                    toHash[sampleFrom] = []
                end
                if fromHash[sampleTo] == nil
                    fromHash[sampleTo] = []
                end

                id += 1
                fromout = fromHash[sampleFrom].length
                fromin = toHash[sampleFrom].length
                toout = fromHash[sampleTo].length
                toin = toHash[sampleTo].length

                fromrate = fromin.to_f / fromout
                torate = toout.to_f / toin

                width = (fromHash[sampleFrom] & toHash[sampleTo]).length

                range = 0
                middle = fromHash[sampleFrom]
                middle.each do |m|
                    if fromHash[m] != nil
                        range += fromHash[m].length
                    end
                end

                intersection = (fromHash[sampleFrom] & fromHash[sampleTo]).length
                union = (fromHash[sampleFrom] | fromHash[sampleTo]).length
                cofrom = intersection.to_f / union

                intersection = (toHash[sampleFrom] & toHash[sampleTo]).length
                union = (toHash[sampleFrom] | toHash[sampleTo]).length
                coto = intersection.to_f / union

                exist = 0

                w << [id,sampleFrom,sampleTo,fromout,fromin,toout,toin,fromrate,torate,width,range,cofrom,coto,exist]

                j += 1
            end
        end
    end
end

puts "end of write result"
